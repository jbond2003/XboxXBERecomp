#include "gpu/shaders.h"
#include "gpu/vertex.h"

#include <algorithm>
#include <bit>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace gpu {

namespace {

std::string F(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return buf;
}

const char* kCommonHead = R"(
struct VOut {
    float4 pos : SV_Position;
    float4 d0 : COLOR0;
    float4 d1 : COLOR1;
    float4 b0 : COLOR2;
    float4 b1 : COLOR3;
    float4 tc0 : TEXCOORD0;
    float4 tc1 : TEXCOORD1;
    float4 tc2 : TEXCOORD2;
    float4 tc3 : TEXCOORD3;
    float fogamt : TEXCOORD4;
};
cbuffer VC : register(b0) { float4 c[192]; };
cbuffer DC : register(b1) {
    float4 surf;
    float4 fogp;
    float4 fogcolor;
    float4 misc;
    float4 pc[18];
    float4 tfactor;
    float4 texsize[4];
    float4 bumpmat[4];
    float4 bumplum[4];
    float4x4 wvp;
)";
const char* kCommonTail = R"(    float4 spread[4];
    float4x4 wv;
    float4x4 wvn;
    float4x4 tm[4];
    float4 mat[5];
    float4 amb;
    float4 lt[56];
    float4 rtshift;
    float4 zparams;
};
)";

const ShaderHooks& Hooks() {
    static const ShaderHooks plain;
    return g_shader_hooks ? *g_shader_hooks : plain;
}

std::string Common() { return kCommonHead + Hooks().Constants() + kCommonTail; }

// vertex programs

const char* kVsHelpers = R"(
float4 mac_mul(float4 a, float4 b) { return (a == 0 || b == 0) ? (float4)0 : a * b; }
int mac_arl(float v) { return (int)floor(v + 1e-3); }
// magnitude clamped between 1 / 2^64 and 2^64, sign kept
float magclamp(float v) { float m = clamp(abs(v), 5.42101e-20, 1.84467e+19); return v < 0 ? -m : m; }
float4 ilu_rcp(float v) { return (1.0 / v).xxxx; }
float4 ilu_rcc(float v) { return magclamp(1.0 / v).xxxx; }
// infinity by its bits: Wine's HLSL compiler (Proton, the Steam Deck) has no 1.#INF
static const float kInf = asfloat(0x7F800000u);
float4 ilu_rsq(float v) { return v == 0 ? (kInf).xxxx : (rsqrt(abs(v))).xxxx; }
// 2^floor(v), fraction, 2^v
float4 ilu_exp(float v) {
    float whole = floor(v);
    return float4(exp2(whole), v - whole, exp2(v), 1);
}
// exponent, mantissa, log2 of |v|; zero logs to negative infinity
float4 ilu_log(float v) {
    float mag = abs(v), l2 = log2(mag), ex = floor(l2);
    return mag == 0 ? float4(-kInf, 1, -kInf, 1) : float4(ex, mag / exp2(ex), l2, 1);
}
// power clamped to 128 less 1/256
float4 ilu_lit(float4 s) {
    const float cap = 127.99609375;
    float diffuse = max(s.x, 0);
    float spec = exp2(clamp(s.w, -cap, cap) * log2(max(s.y, 0)));
    return float4(1, diffuse, diffuse > 0 ? spec : 0, 1);
}
float fogfactor(float d) {
    if (fogp.w == 3) return (fogp.y - d) / (fogp.y - fogp.x);
    if (fogp.w == 1) return exp(-d * fogp.z);
    if (fogp.w == 2) return exp(-(d * fogp.z) * (d * fogp.z));
    return 1;
}
)";

struct Tok {
    const uint32_t* t;
    uint32_t f(int sub, int bit, int len) const { return (t[sub] >> bit) & ((1u << len) - 1); }
};

const char* kMask[16] = {"", "w", "z", "zw", "y", "yw", "yz", "yzw", "x", "xw", "xz", "xzw", "xy", "xyw", "xyz", "xyzw"};
const char* kLanes[5] = {"", "x", "xy", "xyz", "xyzw"};  // oFog takes one lane per mask bit, from x
const char* kOut[16] = {"oPos", "", "", "oD0", "oD1", "oFog", "oPts", "oB0", "oB1", "oT0", "oT1", "oT2", "oT3", "", "", ""};

// operand A 0, B 1, C 2
std::string Src(const Tok& k, int which, uint16_t regs) {
    uint32_t neg, sw[4], reg, mux;
    if (which == 0) {
        neg = k.f(1, 8, 1);
        sw[0] = k.f(1, 6, 2), sw[1] = k.f(1, 4, 2), sw[2] = k.f(1, 2, 2), sw[3] = k.f(1, 0, 2);
        reg = k.f(2, 28, 4), mux = k.f(2, 26, 2);
    } else if (which == 1) {
        neg = k.f(2, 25, 1);
        sw[0] = k.f(2, 23, 2), sw[1] = k.f(2, 21, 2), sw[2] = k.f(2, 19, 2), sw[3] = k.f(2, 17, 2);
        reg = k.f(2, 13, 4), mux = k.f(2, 11, 2);
    } else {
        neg = k.f(2, 10, 1);
        sw[0] = k.f(2, 8, 2), sw[1] = k.f(2, 6, 2), sw[2] = k.f(2, 4, 2), sw[3] = k.f(2, 2, 2);
        reg = k.f(2, 0, 2) << 2 | k.f(3, 30, 2), mux = k.f(3, 28, 2);
        uint32_t ilu = k.f(1, 25, 3);
        if (ilu >= 2 && ilu <= 6) sw[1] = sw[2] = sw[3] = sw[0];  // scalar ops read x
    }
    // mux: 1 temporary (R12 reads oPos), 2 input, 3 constant
    std::string base;
    if (mux == 1) {
        base = reg == 12 ? "oPos" : F("R%u", reg);
    } else if (mux == 2) {
        uint32_t v = k.f(1, 9, 4);
        base = (regs >> v) & 1 ? F("i.v%u", v) : (v == 3 ? "float4(1,1,1,1)" : "float4(0,0,0,1)");
    } else {
        uint32_t cr = k.f(1, 13, 8);
        base = k.f(3, 1, 1) ? F("c[clamp(a0 + %u, 0, 191)]", cr) : F("c[%u]", cr);
    }
    const char* xyzw = "xyzw";
    std::string s = F("(%s(%s).%c%c%c%c)", neg ? "-" : "", base.c_str(), xyzw[sw[0]], xyzw[sw[1]], xyzw[sw[2]], xyzw[sw[3]]);
    return s;
}

// MAC opcodes from 1: mov mul add mad dp3 dph dp4 dst min max slt sge arl
std::string MacExpr(uint32_t op, const std::string& a, const std::string& b, const std::string& c) {
    switch (op) {
        case 1: return a;
        case 2: return "mac_mul(" + a + ", " + b + ")";
        case 3: return "(" + a + " + " + c + ")";
        case 4: return "(mac_mul(" + a + ", " + b + ") + " + c + ")";
        case 5: return "dot(" + a + ".xyz, " + b + ".xyz).xxxx";
        case 6: return "dot(float4(" + a + ".xyz, 1), " + b + ").xxxx";
        case 7: return "dot(" + a + ", " + b + ").xxxx";
        case 8: return "float4(1, " + a + ".y * " + b + ".y, " + a + ".z, " + b + ".w)";
        case 9: return "min(" + a + ", " + b + ")";
        case 10: return "max(" + a + ", " + b + ")";
        case 11: return "(float4)(" + a + " < " + b + ")";
        case 12: return "(float4)(" + a + " >= " + b + ")";
        case 13: return a;
        default: return "(float4)0";
    }
}

// ILU opcodes from 1: mov rcp rcc rsq exp log lit
std::string IluExpr(uint32_t op, const std::string& c) {
    switch (op) {
        case 1: return c;
        case 2: return "ilu_rcp(" + c + ".x)";
        case 3: return "ilu_rcc(" + c + ".x)";
        case 4: return "ilu_rsq(" + c + ".x)";
        case 5: return "ilu_exp(" + c + ".x)";
        case 6: return "ilu_log(" + c + ".x)";
        case 7: return "ilu_lit(" + c + ")";
        default: return "(float4)0";
    }
}

std::string RegDest(uint32_t r) { return r == 12 ? "oPos" : F("R%u", r); }

std::string VsInputStruct(uint16_t regs) {
    std::string s = "struct VIn {\n";
    for (int r = 0; r < 16; r++)
        if ((regs >> r) & 1) s += F("    float4 v%d : ATTR%d;\n", r, r);
    if (!regs) s += "    uint id : SV_VertexID;\n";
    return s + "};\n";
}

// the input struct and the head of main: elements read raw are finished into float4 i.vN first
std::string VsInputs(uint16_t regs, uint64_t decode) {
    if (!decode) return VsInputStruct(regs) + "VOut main(VIn i) {\n";
    std::string raw = "struct VRaw {\n", in = "struct VIn {\n", body;
    bool packed = false;
    for (int r = 0; r < 16; r++) {
        if (!((regs >> r) & 1)) continue;
        uint32_t d = uint32_t(decode >> (r * 4)) & 15;
        const char* type = d == kDecodeInt || d == kDecodeIntW1 ? "int4" : d == kDecodePacked ? "uint" : "float4";
        raw += F("    %s v%d : ATTR%d;\n", type, r, r);
        in += F("    float4 v%d;\n", r);
        packed |= d == kDecodePacked;
        switch (d) {
            case kDecodeW1: body += F("    i.v%d = float4(ri.v%d.xyz, 1);\n", r, r); break;
            case kDecodeInt: body += F("    i.v%d = float4(ri.v%d);\n", r, r); break;
            case kDecodeIntW1: body += F("    i.v%d = float4(float3(ri.v%d.xyz), 1);\n", r, r); break;
            case kDecodePacked: body += F("    i.v%d = UnpackNp3(ri.v%d);\n", r, r); break;
            case kDecodeFloat2H: body += F("    i.v%d = float4(ri.v%d.xy, 0, ri.v%d.z);\n", r, r, r); break;
            default: body += F("    i.v%d = ri.v%d;\n", r, r); break;
        }
    }
    std::string s = raw + "};\n" + in + "};\n";
    if (packed)
        s += "float4 UnpackNp3(uint u) {\n"
             "    int3 v = int3(int(u << 21) >> 21, int(u << 10) >> 21, int(u) >> 22);\n"
             "    return float4(max(float3(v) / float3(1023, 1023, 511), -1), 1);\n"
             "}\n";
    return s + "VOut main(VRaw ri) {\n    VIn i;\n" + body;
}

const char* kVsEpilogue = R"(
    VOut o;
    precise float4 p = oPos;
    p.w = magclamp(p.w);
    precise float4 q;
    // full screen passes drawn half a pixel up-left for the console's integer pixel centers stop half a pixel short
    // of the far edges on D3D12's, which upscaled leaves the last host columns and rows undrawn
    // surf.w set: a point, which stays put since pulled out it would leave the target
    if (surf.w == 0 && abs(p.x - (surf.x - 0.5)) < 1e-3) p.x = surf.x;
    if (surf.w == 0 && abs(p.y - (surf.y - 0.5)) < 1e-3) p.y = surf.y;
    q.x = p.x * 2 / surf.x - 1;
    q.y = 1 - p.y * 2 / surf.y;
    q.z = p.z * surf.z;
    q.xyz *= p.w;
    q.w = p.w;
    o.pos = q;
    o.d0 = saturate(oD0);
    o.d1 = saturate(oD1);
    o.b0 = saturate(oB0);
    o.b1 = saturate(oB1);
    o.tc0 = oT0;
    o.tc1 = oT1;
    o.tc2 = oT2;
    o.tc3 = oT3;
    o.fogamt = fogfactor(oFog.x);
    return o;
}
)";

const char* kVsPrologue = R"(
    int a0 = 0;
    // precise: EQUAL depth passes need bit exact positions
    precise float4 R0 = 0, R1 = 0, R2 = 0, R3 = 0, R4 = 0, R5 = 0, R6 = 0, R7 = 0, R8 = 0, R9 = 0, R10 = 0, R11 = 0;
    precise float4 oPos = 0;
    float4 oD0 = float4(0, 0, 0, 1), oD1 = 0, oB0 = float4(0, 0, 0, 1), oB1 = 0, oFog = 0, oPts = 1;
    float4 oT0 = float4(0, 0, 0, 1), oT1 = float4(0, 0, 0, 1), oT2 = float4(0, 0, 0, 1), oT3 = float4(0, 0, 0, 1);
)";

// combiners

struct In {
    uint32_t reg, chan, mod;
};
In ParseIn(uint32_t v) { return {v & 0xF, v & 0x10, v & 0xE0}; }

struct Ps {
    const PixelKey& k;
    int stage = 0;
    std::string e, f;  // final combiner E F for EF_PROD
    bool clamp_sum = false, inv_v1 = false, inv_r0 = false;
    std::string code;

    std::string Var(uint32_t reg) {
        uint32_t flags = k.count >> 8;  // PS_COMBINERCOUNT flags: 0x10 unique C0 per stage, 0x100 unique C1
        switch (reg) {
            case 0: return "float4(0,0,0,0)";
            case 1: return F("pc[%d]", stage == 8 ? 16 : ((flags & 0x10) ? stage * 2 : 0));
            case 2: return F("pc[%d]", stage == 8 ? 17 : ((flags & 0x100) ? stage * 2 + 1 : 1));
            case 3: return "fogin";
            case 4: return "v0";
            case 5: return "v1";
            case 8: return "t0";
            case 9: return "t1";
            case 10: return "t2";
            case 11: return "t3";
            case 12: return "r0";
            case 13: return "r1";
            case 14:
                return F("%s(float4(%s.rgb + %s.rgb, 0))", clamp_sum ? "saturate" : "", inv_v1 ? "(1 - v1)" : "v1",
                         inv_r0 ? "(1 - r0)" : "r0");
            case 15: return "float4(" + e + " * " + f + ", 0)";
            default: return "float4(0,0,0,0)";
        }
    }

    std::string Input(In in, bool alpha) {
        std::string r = Var(in.reg);
        if (!alpha) r += in.chan ? ".aaa" : ".rgb";
        else r += in.chan ? ".a" : ".b";
        switch (in.mod) {
            // PS_INPUTMAPPING
            case 0x00: return "max(" + r + ", 0)";
            case 0x20: return "(1 - saturate(" + r + "))";
            case 0x40: return "(2 * max(" + r + ", 0) - 1)";
            case 0x60: return "(-2 * max(" + r + ", 0) + 1)";
            case 0x80: return "(max(" + r + ", 0) - 0.5)";
            case 0xA0: return "(-max(" + r + ", 0) + 0.5)";
            case 0xC0: return r;
            default: return "-" + r;
        }
    }

    static std::string Map(const std::string& v, uint32_t m) {
        switch (m) {
            // PS_COMBINEROUTPUT bias and shift
            case 0x08: return "(" + v + " - 0.5)";
            case 0x10: return "(" + v + " * 2)";
            case 0x18: return "((" + v + " - 0.5) * 2)";
            case 0x20: return "(" + v + " * 4)";
            case 0x30: return "(" + v + " / 2)";
            default: return v;
        }
    }

    static const char* Dest(uint32_t reg) {
        switch (reg) {
            case 4: return "v0";
            case 5: return "v1";
            case 8: return "t0";
            case 9: return "t1";
            case 10: return "t2";
            case 11: return "t3";
            case 12: return "r0";
            case 13: return "r1";
            default: return nullptr;
        }
    }

    // rgb and alpha halves read the registers before either writes, so the writes come back to the caller
    std::string Stage(uint32_t inputs, uint32_t output, bool alpha) {
        In d = ParseIn(inputs & 0xFF), c = ParseIn((inputs >> 8) & 0xFF), b = ParseIn((inputs >> 16) & 0xFF),
           a = ParseIn(inputs >> 24);
        // fl: 1 cd dot, 2 ab dot, 4 mux, 0x38 mapping, 0x40 and 0x80 blue to alpha
        uint32_t cd_dst = output & 0xF, ab_dst = (output >> 4) & 0xF, sum_dst = (output >> 8) & 0xF, fl = output >> 12;
        const char* m = alpha ? "a" : "rgb";
        const char* T = alpha ? "float" : "float3";
        const char* n = alpha ? "a" : "c";
        std::string A = Input(a, alpha), B = Input(b, alpha), C = Input(c, alpha), D = Input(d, alpha);
        std::string ab = (fl & 2) ? "dot(" + A + ", " + B + ")" : "(" + A + " * " + B + ")";
        std::string cd = (fl & 1) ? "dot(" + C + ", " + D + ")" : "(" + C + " * " + D + ")";
        uint32_t map = fl & 0x38;
        code += F("%s ab%s = clamp((%s)%s, -1, 1);\n", T, n, T, Map(ab, map).c_str());
        code += F("%s cd%s = clamp((%s)%s, -1, 1);\n", T, n, T, Map(cd, map).c_str());
        std::string sum = (fl & 4) ? F("((%s) ? (%s)%s : (%s)%s)",
                                       ((k.count >> 8) & 1) ? "r0.a >= 0.5" : "(uint(r0.a * 255) & 1) == 1", T,
                                       cd.c_str(), T, ab.c_str())
                                   : "(" + ab + " + " + cd + ")";
        code += F("%s ms%s = clamp((%s)%s, -1, 1);\n", T, n, T, Map(sum, map).c_str());
        std::string post;
        if (const char* r = Dest(ab_dst)) {
            post += F("%s.%s = ab%s;\n", r, m, n);
            if (!alpha && (fl & 0x80)) post += F("%s.a = abc.b;\n", r);
        }
        if (const char* r = Dest(cd_dst)) {
            post += F("%s.%s = cd%s;\n", r, m, n);
            if (!alpha && (fl & 0x40)) post += F("%s.a = cdc.b;\n", r);
        }
        if (const char* r = Dest(sum_dst)) post += F("%s.%s = ms%s;\n", r, m, n);
        return post;
    }
};

// a dot stage's operand: the source stage's rgb, raw or read back as signed bytes
std::string DotInput(uint32_t map, uint32_t src) {
    switch (map) {
        case 1: return F("sbyte(t%u.rgb, 128, 128, 127)", src);        // offset by 128
        case 2: return F("sbyte(t%u.rgb, -0.5, 255.5, 127.5)", src);  // two's complement, (2b + 1) / 255
        case 3: return F("sbyte(t%u.rgb, 0, 256, 127)", src);         // two's complement
        default: return F("t%u.rgb", src);
    }
}

// stage i's coordinates as the title wrote them, over 0..1. The console draws half a pixel right and down of the
// host unless a program writes positions raw: a render target is read where the console's offset puts it
std::string TexUv(const TexInfo& t, int i, const std::string& uv) {
    std::string n = "(" + uv + ")";
    if (t.rect) n += F(" / texsize[%d].xy", i);
    n += F(" + texsize[%d].zw", i);
    if (!(t.flags & kTexTarget)) return n;
    std::string step = "ddx(" + n + ") + ddy(" + n + ")";
    return n + F(" + rtshift[%d] * (", i) + step + ")";
}

// texture shader stage i, per its mode
std::string TexFetch(const PixelKey& k, int i) {
    uint32_t mode = (k.tex_modes >> (i * 5)) & 0x1F;
    const TexInfo& t = k.tex[i];
    uint32_t in = i == 1 ? 0 : i == 2 ? (k.input_texture >> 16) & 0xF : (k.input_texture >> 20) & 0xF;
    uint32_t dm = i == 0 ? 0 : (k.dot_mapping >> ((i - 1) * 4)) & 0xF;
    const std::string tc = F("tc%d", i);
    const char* q = tc.c_str();
    auto put = [&](const std::string& v) { return F("float4 t%d = %s;\n", i, v.c_str()); };
    // dpN, or dpNn for stage N + 1's product taken early
    auto dp = [&](int n, int coord, uint32_t map, uint32_t src) {
        return F("float dp%d%s = dot(tc%d.xyz, %s);\n", n, coord == n ? "" : "n", coord, DotInput(map, src).c_str());
    };
    auto tex2d = [&](const std::string& uv) {
        if (t.flags & kTexWide) return F("spreadtap(tex%d, smp%d, %s, spread[%d], %s)", i, i, uv.c_str(), i, Hooks().ReadScale(t).c_str());
        if (t.flags & kTexCrisp) return F("crisp(tex%d.Sample(smp%d, %s))", i, i, uv.c_str());
        return F("tex%d.Sample(smp%d, %s)", i, i, uv.c_str());
    };
    auto cube = [&](const std::string& dir) {
        std::string c = Hooks().Cube(t, i, dir);
        return c.empty() ? F("tex%d.Sample(smp%d, %s)", i, i, dir.c_str()) : c;
    };
    auto sample2 = [&](const std::string& uv) {
        if (t.kind == kTex2D) return tex2d(TexUv(t, i, uv));
        if (t.kind == kTexCube) return cube("float3(" + uv + ", 1)");
        if (t.kind == kTex3D) return F("tex%d.Sample(smp%d, float3(%s, 0))", i, i, uv.c_str());
        return std::string("float4(0,0,0,1)");
    };
    auto sample3 = [&](const std::string& uvw) {
        if (t.kind == kTex2D) return tex2d(TexUv(t, i, "(" + uvw + ").xy"));
        if (t.kind == kTexCube) return cube(uvw);
        if (t.kind == kTex3D) return F("tex%d.Sample(smp%d, %s)", i, i, uvw.c_str());
        return std::string("float4(0,0,0,1)");
    };
    auto depth = [&](const std::string& ref) {
        std::string uv = TexUv(t, i, F("%s.xy / %s.w", q, q));
        std::string rest = F(", %s, %d, i.pos.xy, texsize[%d].xy)", ref.c_str(), t.shadow, i);
        return put(F("shadow(tex%d, smp%d, ", i, i) + uv + rest);
    };
    auto sun = [&](const std::string& p) {
        std::string light = (t.flags & kTexSunLight) && i > 0 ? F("t%d", i - 1) : std::string("float4(1, 1, 1, 1)");
        std::string uv = TexUv(t, i, F("(%s).xy", p.c_str()));
        return put(F("sunshadow(tex%d, smp%d, %s, %s, %s)", i, i, p.c_str(), uv.c_str(), light.c_str()));
    };
    if (t.flags & kTexSun) return mode == 1 ? sun(F("%s.xyz / %s.w", q, q)) : put("float4(0, 0, 0, 1)");
    std::string s;
    switch (mode) {
        // PS_TEXTUREMODES: 1 project2d, 2 project3d, 3 cubemap, 4 passthru, 5 clipplane, 6 and 7 bumpenvmap, 9 dot_st, 0xA dot_zw, 0xD dot_str_3d, 0xE dot_str_cube, 0xF dpndnt_ar, 0x10 dpndnt_gb, 0x11 dotproduct
        case 0: return put("float4(0,0,0,1)");
        case 1: return t.shadow ? depth("0") : put(sample2(F("%s.xy / %s.w", q, q)));
        case 2: return t.shadow ? depth(F("%s.z / %s.w", q, q)) : put(sample3(F("%s.xyz / %s.w", q, q)));
        case 3: return put(sample3(tc + ".xyz"));
        case 4: return put(tc);
        case 5:  // clip plane: a set compare_mode bit kills at >= 0, a clear one below 0
            s = put("0");
            for (int j = 0; j < 4; j++) s += F("if (%s.%c %s 0) discard;\n", q, "xyzw"[j], ((k.compare_mode >> (i * 4 + j)) & 1) ? ">=" : "<");
            return s;
        case 6:
        case 7:  // bump env map: signed du dv through the 2x2 matrix, 7 scales by luminance
            s = F("float2 duv%d = sbyte(t%u.bgr, 0, 256, 127).xy;\n", i, in);
            s += put(sample2(F("%s.xy + float2(dot(bumpmat[%d].xz, duv%d), dot(bumpmat[%d].yw, duv%d))", q, i, i, i, i)));
            if (mode == 7) s += F("t%d *= bumplum[%d].x * t%u.r + bumplum[%d].y;\n", i, i, in, i);
            return s;
        case 9: return dp(i, i, dm, in) + put(sample2(F("float2(dp%d, dp%d)", i - 1, i)));
        case 0xA:
        case 0x11: return dp(i, i, dm, in) + put("0");
        case 0xB:  // reflect diffuse at stage 2: stage 3's product early for the normal
            s = dp(i, i, dm, in) + dp(i, i + 1, (k.dot_mapping >> 8) & 0xF, (k.input_texture >> 20) & 0xF);
            return s + put(sample3(F("float3(dp%d, dp%d, dp%dn)", i - 1, i, i)));
        case 0xC:  // reflect specular at stage 3: eye from the three w
            s = dp(i, i, dm, in);
            s += F("float3 nrm%d = float3(dp%d, dp%d, dp%d), eye%d = float3(tc%d.w, tc%d.w, tc%d.w);\n", i, i - 2, i - 1, i, i, i - 2,
                   i - 1, i);
            s += F("float3 refl%d = 2 * nrm%d * dot(nrm%d, eye%d) / dot(nrm%d, nrm%d) - eye%d;\n", i, i, i, i, i, i, i);
            return s + put(sample3(F("refl%d", i)));
        case 0xD:
        case 0xE: return dp(i, i, dm, in) + put(sample3(F("float3(dp%d, dp%d, dp%d)", i - 2, i - 1, i)));
        case 0xF: return put(sample2(F("t%u.ar", in)));
        case 0x10: return put(sample2(F("t%u.gb", in)));
        default: return put("0");
    }
}

const char* kPsSbyte = R"(
// unorm to byte b, less lo under 128 or hi from 128, over d
float3 sbyte(float3 u, float lo, float hi, float d) { float3 b = u * 255; return (b - (b >= 128 ? hi : lo)) / d; }
)";
const char* kPsTail = R"(// A silhouette drawn into a large target and laid on the world from far closer than it was drawn (a shadow picture):
// read between its texels the edge is a slope a texel wide, a blur many pixels across on the ground at the player's
// feet. The slope is made steep about its middle: a fifth of a texel, never less than a pixel, so far from the eye,
// where a texel is under a pixel, the alpha is as it was read.
float4 crisp(float4 t) {
    float w = max(fwidth(t.a), 0.2);
    t.a = saturate((t.a - 0.5) / w + 0.5);
    return t;
}
float4 spreadtap(Texture2D t, SamplerState s, float2 uv, float4 w, float m) {
    float4 c = 0;
    int n = (int)w.z;
    for (int y = 0; y < n; y++)
        for (int x = 0; x < n; x++) {
            float4 v = t.SampleLevel(s, uv + w.xy * (float2(x, y) - (n - 1) * 0.5), 0);
            c += float4(saturate(v.rgb * m), v.a);
        }
    return c / (n * n);
}
void alphatest(float a) {
    int f = (int)misc.y;
    int v = (int)round(a * 255), r = (int)misc.x;
    bool ok = f == 0 ? false : f == 1 ? v < r : f == 2 ? v == r : f == 3 ? v <= r : f == 4 ? v > r : f == 5 ? v != r : f == 6 ? v >= r : true;
    if (!ok) discard;
}
)";

const char* kPsSun = R"(// The shadow picture read by its depth (kTexSun, d3d::SunShadows): p the stage's coordinates over their w, at where
// they read the picture. The picture's u and v are sun[2] . (p, 1) and sun[3] . (p, 1): the stage's own, but in a frame
// that reads the picture another camera left (d3d::kSunStill) that picture's. A point is in shadow where the picture
// holds something nearer its camera: its own depth is sun[0] . (p, 1), 0 at the picture's near plane and 1 at its far
// one, and the picture holds 1 less that (depth is stored reversed). A box 3 texels across laid just where the point
// is: 4 gathers, 16 texels, the outer ones weighed by how much of them the box covers (so the shadow's edge moves
// evenly with the point, no step from texel to texel), each gather tested along the receiver's own plane (its slope
// in depth per uv from the screen derivatives, at most sun[1].w) less the bias sun[1].z.
// light: the surface's baked light (kTexSunLight), darkened only down to the shade sun[1].y. Handed on as the
// silhouette picture's alpha (black, alpha the shadow) times sun[1].x.
// Only in the shaders that read it (PixelHlsl), with its constants at the end of the draw's: every other shader's
// text, and so its place in the cache of compiled ones, is as it was.

// kx, ky: the weights of a gather's left and right columns, of its upper and lower rows (v grows downwards)
float sunopen(Texture2D t, SamplerState s, float2 at, float limit, float2 kx, float2 ky) {
    float4 g = t.GatherRed(s, at);  // left lower, right lower, right upper, left upper
    return dot(step(g, limit), float4(kx.x * ky.y, kx.y * ky.y, kx.y * ky.x, kx.x * ky.x));
}
float4 sunshadow(Texture2D t, SamplerState s, float3 p, float2 at, float4 light) {
    float4 q = float4(p, 1);
    float z = dot(q, sun[0]);
    float2 uv = float2(dot(q, sun[2]), dot(q, sun[3])) + (at - p.xy);
    float2 ux = ddx(uv), uy = ddy(uv);
    float zx = ddx(z), zy = ddy(z);
    if (any(uv <= 0) || any(uv >= 1) || !(z > 0 && z < 1)) return float4(0, 0, 0, 0);
    float w, h;
    t.GetDimensions(w, h);
    float2 size = float2(w, h), texel = 1 / size;
    float det = ux.x * uy.y - ux.y * uy.x;
    float2 slope = float2(0, 0);
    if (abs(det) > 1e-24) slope = float2(uy.y * zx - ux.y * zy, ux.x * zy - uy.x * zx) / det;
    slope = clamp(slope, -sun[1].w, sun[1].w);
    // reversed: lit where the picture holds no more than 1 - (z - bias), a texel's slope taken off for its rounding
    float limit = 1 - z + sun[1].z + dot(abs(slope), texel);
    // the texel the box's middle is in or just past is corner; the box covers corner - 1 to corner + 2, the first by
    // 1 - part and the last by part. A gather at a texel's corner takes the four texels that meet there.
    float2 first = uv * size - 0.5, corner = floor(first), part = first - corner;
    float4 kx = float4(1 - part.x, 1, 1, part.x), ky = float4(1 - part.y, 1, 1, part.y);
    float2 a = corner * texel, b = a + 2 * texel;
    float open = sunopen(t, s, a, limit - dot(slope, a - uv), kx.xy, ky.xy);
    open += sunopen(t, s, float2(b.x, a.y), limit - dot(slope, float2(b.x, a.y) - uv), kx.zw, ky.xy);
    open += sunopen(t, s, float2(a.x, b.y), limit - dot(slope, float2(a.x, b.y) - uv), kx.xy, ky.zw);
    open += sunopen(t, s, b, limit - dot(slope, b - uv), kx.zw, ky.zw);
    // nothing at the edges of the square the shadows end at (sun[4]: where it lies from the picture's own, how far
    // inside) and at the picture's far end, coming in over a twentieth of it: no line where it stops
    float2 e = uv + sun[4].xy;
    e = min(e, 1 - e) - sun[4].z;
    float dark = (1 - open / 9) * saturate(min(min(e.x, e.y), 1 - z) * 20);
    if (sun[1].y > 0) dark *= saturate(1 - sun[1].y / max(dot(light.rgb, float3(1, 1, 1)) / 3, 1e-3));
    return float4(0, 0, 0, dark * sun[1].x);
}
)";
// kCommonTail's end, and the same with sunshadow's constants (DrawConstants::sun)
const char* kConstantsEnd = "    float4 zparams;\n};\n";
const char* kConstantsEndSun = "    float4 zparams;\n    float4 sun[5];\n};\n";

std::string PsHelpers() { return kPsSbyte + Hooks().Shadow() + kPsTail; }

// fixed function texture stage arguments, D3DTA
std::string FfArg(uint32_t a, int stage, bool alpha) {
    std::string v;
    switch (a & 0xF) {
        case 0: v = "v0"; break;
        case 1: v = "cur"; break;
        case 2: v = F("t%d", stage); break;
        case 3: v = "tfactor"; break;
        case 4: v = "v1"; break;
        case 5: v = "tmp"; break;
        default: v = "cur"; break;
    }
    if (a & 0x20) v = "(" + v + ").aaaa";
    if (a & 0x10) v = "(1 - " + v + ")";
    return "(" + v + ")" + (alpha ? ".a" : ".rgb");
}

// D3DTOP
std::string FfOp(uint32_t op, const std::string& a1, const std::string& a2, const std::string& a0, int stage, bool alpha) {
    switch (op) {
        case 2: return a1;
        case 3: return a2;
        case 4: return a1 + " * " + a2;
        case 5: return a1 + " * " + a2 + " * 2";
        case 6: return a1 + " * " + a2 + " * 4";
        case 7: return a1 + " + " + a2;
        case 8: return a1 + " + " + a2 + " - 0.5";
        case 9: return "(" + a1 + " + " + a2 + " - 0.5) * 2";
        case 10: return a1 + " - " + a2;
        case 11: return a1 + " + " + a2 + " * (1 - " + a1 + ")";
        case 12: return "lerp(" + a2 + ", " + a1 + ", v0.a)";
        case 13: return "lerp(" + a2 + ", " + a1 + ", cur.a)";
        case 14: return F("lerp(%s, %s, t%d.a)", a2.c_str(), a1.c_str(), stage);
        case 15: return "lerp(" + a2 + ", " + a1 + ", tfactor.a)";
        case 16: return F("%s + %s * (1 - t%d.a)", a1.c_str(), a2.c_str(), stage);
        case 18: return alpha ? a1 : a1 + " + " + a2 + " * " + a1.substr(0, a1.size() - 4) + ".a";
        case 19: return alpha ? a1 : a1 + " * " + a2 + " + " + a1.substr(0, a1.size() - 4) + ".a";
        case 20: return alpha ? a1 : a1 + " + " + a2 + " * (1 - " + a1.substr(0, a1.size() - 4) + ".a)";
        case 21: return alpha ? a1 : "(1 - " + a1 + ") * " + a2 + " + " + a1.substr(0, a1.size() - 4) + ".a";
        case 22: return F("saturate(dot((%s - 0.5) * 2, (%s - 0.5) * 2))", alpha ? ("float3(" + a1 + ",0,0)").c_str() : a1.c_str(),
                          alpha ? ("float3(" + a2 + ",0,0)").c_str() : a2.c_str());
        case 23: return a0 + " + " + a1 + " * " + a2;
        case 24: return "lerp(" + a2 + ", " + a1 + ", " + a0 + ")";
        default: return alpha ? "cur.a" : "cur.rgb";
    }
}

}  // namespace

bool PixelKey::operator==(const PixelKey& o) const { return std::memcmp(this, &o, sizeof(*this)) == 0; }

std::string ShaderHooks::Constants() const { return "    float4 ext;\n"; }

std::string ShaderHooks::Shadow() const {
    std::string r = g_reversed_depth ? "1 - clamp(z, 0, mx) / mx" : "clamp(z, 0, mx) / mx";
    return R"(// the sampler compares, a linear filter blends the four results like the console does
float4 shadow(Texture2D<float> t, SamplerComparisonState s, float2 uv, float z, int fmt, float2 px, float2 size) {
    float mx = fmt == 2 ? 65535.0 : 16777215.0;
    return t.SampleCmpLevelZero(s, uv, )" + r + R"().xxxx;
}
)";
}

std::string ShaderHooks::Output() const { return "outc.rgb = saturate(outc.rgb);\n"; }

uint64_t Hash(const void* p, size_t n, uint64_t h) {
    // 16 bytes a step through a 64 x 64 -> 128 bit multiply: it runs over textures every frame and keys every draw
    constexpr uint64_t k0 = 0x9E3779B97F4A7C15ull, k1 = 0xE7037ED1A0B428DBull;
    auto mix = [](uint64_t a, uint64_t b) {
        unsigned __int128 r = static_cast<unsigned __int128>(a) * b;
        return uint64_t(r) ^ uint64_t(r >> 64);
    };
    const uint8_t* b = static_cast<const uint8_t*>(p);
    uint64_t seed = h ^ k0;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint64_t x, y;
        std::memcpy(&x, b + i, 8);
        std::memcpy(&y, b + i + 8, 8);
        seed = mix(x ^ k1, y ^ seed);
    }
    if (size_t r = n - i) {
        uint64_t x = 0, y = 0;
        if (r >= 8) {
            std::memcpy(&x, b + i, 8);
            std::memcpy(&y, b + n - 8, 8);
        } else {
            std::memcpy(&x, b + i, r);
        }
        seed = mix(x ^ k1, y ^ seed ^ r);
    }
    return mix(seed ^ k0, uint64_t(n) ^ k1);
}

namespace {

// D3DTOP and D3DTA values PixelKey::ff holds
constexpr uint32_t kTopSelectArg1 = 2;
constexpr uint32_t kTopSelectArg2 = 3;
constexpr uint32_t kTopBlendTextureAlpha = 14;
constexpr uint32_t kTopBlendTextureAlphaPm = 16;
constexpr uint32_t kTopModulateAlphaAddColor = 18;
constexpr uint32_t kTopModulateInvColorAddAlpha = 21;
constexpr uint32_t kTopMultiplyAdd = 23;
constexpr uint32_t kTopLerp = 24;
constexpr uint32_t kTaCurrent = 1;
constexpr uint32_t kTaTexture = 2;

// args in PixelKey::ff order: arg1, arg2, arg0
bool ReadsTexture(uint32_t op, const uint32_t* args, bool alpha) {
    if (op == kTopBlendTextureAlpha || op == kTopBlendTextureAlphaPm) return true;
    bool alpha_arg1_only = alpha && op >= kTopModulateAlphaAddColor && op <= kTopModulateInvColorAddAlpha;
    bool uses_arg1 = op != kTopSelectArg2;
    bool uses_arg2 = op != kTopSelectArg1 && !alpha_arg1_only;
    bool uses_arg0 = op == kTopMultiplyAdd || op == kTopLerp;
    auto texture = [](uint32_t a) { return (a & 0xF) == kTaTexture; };
    return (uses_arg1 && texture(args[0])) || (uses_arg2 && texture(args[1])) || (uses_arg0 && texture(args[2]));
}

// the XDK's combiner builder turns a stage op that reads a missing texture into a pass of the current colour
void PassMissingTexture(uint32_t* f) {
    if (f[0] > 1 && ReadsTexture(f[0], f + 1, false)) {
        f[0] = kTopSelectArg1;
        f[1] = kTaCurrent;
    }
    if (f[4] > 1 && ReadsTexture(f[4], f + 5, true)) {
        f[4] = kTopSelectArg1;
        f[5] = kTaCurrent;
    }
}

}  // namespace

PixelKey MakePixelKey(const State& s, const TexInfo tex[4]) {
    PixelKey k;
    std::memset(&k, 0, sizeof(k));
    for (int i = 0; i < 4; i++) {
        k.tex[i] = tex[i];
        if (s.tss[size_t(i)][kTssAlphaKill] == 4) k.alphakill |= 1u << i;  // D3DTALPHAKILL_ENABLE
    }
    k.flat = s.simple[kMShadeMode >> 2] == 0x1D00;  // D3DSHADE_FLAT
    k.two_sided = s.rs[kRsTwoSidedLighting] != 0;
    k.alphatest = s.simple[kMAlphaTestEnable >> 2] != 0;
    if (s.ps) {
        const uint32_t* d = s.ps->def;
        k.programmable = 1;
        // D3DPIXELSHADERDEF dwords: alpha inputs 0, final 8 and 9, alpha outputs 26, rgb inputs 34, compare mode 42, rgb outputs 45, count 53, dot mapping 55, input texture 56
        k.count = d[53];
        for (int i = 0; i < 8; i++) {
            k.a_in[i] = d[i];
            k.a_out[i] = d[26 + i];
            k.rgb_in[i] = d[34 + i];
            k.rgb_out[i] = d[45 + i];
        }
        k.final_abcd = d[8];
        k.final_efg = d[9];
        // the stage modes follow the bound textures, as D3D's lazy shader stage setup does
        uint32_t modes = s.rs[kRsPsTextureModes];
        for (int i = 0; i < 4; i++) {
            uint32_t m = (modes >> (i * 5)) & 0x1F;
            const TexInfo& t = tex[i];
            if (!t.kind && m != 4 && m != 5 && m != 0xA && m != 0x11) m = 0;
            else if (m >= 1 && m <= 3) m = t.kind == kTexCube ? 3 : (t.kind == kTex3D || t.shadow) ? 2 : 1;
            else if (m == 0xD || m == 0xE) m = t.kind == kTexCube ? 0xE : 0xD;
            k.tex_modes |= m << (i * 5);
        }
        k.dot_mapping = d[55];
        k.input_texture = d[56];
        k.compare_mode = d[42];
        return k;
    }
    // the console's stages divide by q; a vertex program's q is its own
    bool program = (s.vs_handle & 1) && s.vs && !s.vs->program.empty();
    for (int i = 0; i < 4; i++) {
        const auto& t = s.tss[size_t(i)];
        uint32_t v[9] = {t[kTssColorOp], t[kTssColorArg1], t[kTssColorArg2], t[kTssColorArg0], t[kTssAlphaOp],
                         t[kTssAlphaArg1],  t[kTssAlphaArg2], t[kTssAlphaArg0], t[kTssResultArg]};
        std::memcpy(k.ff[i], v, sizeof(v));
        if (!tex[i].kind) PassMissingTexture(k.ff[i]);
        if (program || (t[kTssTextureTransformFlags] & 0x100)) k.proj |= 1u << i;  // D3DTTFF_PROJECTED
        if (t[kTssColorOp] <= 1) break;
    }
    return k;
}

constexpr uint32_t kMuxConstant = 3;
constexpr uint32_t kViewportOffsetConstant = 59;  // c-37: constants start at c-96

bool ReadsViewportOffset(const std::vector<uint32_t>& prog) {
    for (size_t n = 0; n + 4 <= prog.size(); n += 4) {
        Tok k{&prog[n]};
        // an instruction reads at most one constant, through any of its operands
        bool constant = k.f(2, 26, 2) == kMuxConstant || k.f(2, 11, 2) == kMuxConstant || k.f(3, 28, 2) == kMuxConstant;
        bool relative = k.f(3, 1, 1);
        if (constant && !relative && k.f(1, 13, 8) == kViewportOffsetConstant) return true;
    }
    return false;
}

std::string VertexProgramHlsl(const std::vector<uint32_t>& prog, uint16_t regs, uint64_t decode) {
    std::string s = Common();
    s += kVsHelpers;
    s += VsInputs(regs, decode);
    s += kVsPrologue;
    for (size_t n = 0; n + 4 <= prog.size(); n += 4) {
        Tok k{&prog[n]};
        uint32_t mac = k.f(1, 21, 4), ilu = k.f(1, 25, 3);
        if (mac || ilu) {
            std::string a = Src(k, 0, regs), b = Src(k, 1, regs), c = Src(k, 2, regs);
            s += "    {\n";
            if (mac) s += "        precise float4 mr = " + MacExpr(mac, a, b, c) + ";\n";
            if (ilu) s += "        precise float4 ir = " + IluExpr(ilu, c) + ";\n";
            uint32_t r = k.f(3, 20, 4), omask = k.f(3, 12, 4);
            if (mac == 13) {
                s += "        a0 = mac_arl(mr.x);\n";
            } else if (mac) {
                uint32_t m = k.f(3, 24, 4);
                if (ilu && r == 1) m = 0;
                if (m && r <= 12) s += F("        %s.%s = mr.%s;\n", RegDest(r).c_str(), kMask[m], kMask[m]);
            }
            if (ilu) {
                uint32_t m = k.f(3, 16, 4), ir = mac ? 1 : r;
                if (m && ir <= 12) s += F("        %s.%s = ir.%s;\n", RegDest(ir).c_str(), kMask[m], kMask[m]);
            }
            bool use_ilu = k.f(3, 2, 1) == 1;
            if (omask && k.f(3, 11, 1) && ((use_ilu && ilu) || (!use_ilu && mac && mac != 13))) {
                uint32_t o = k.f(3, 3, 8) & 0xF;
                const char* src = use_ilu ? "ir" : "mr";
                if (kOut[o][0]) {
                    const char* msk = o == 5 ? kLanes[std::popcount(omask)] : kMask[omask];
                    s += F("        %s.%s = %s.%s;\n", kOut[o], msk, src, msk);
                }
            }
            s += "    }\n";
        }
        if (k.f(3, 0, 1)) break;  // last instruction
    }
    s += kVsEpilogue;
    return s;
}

FixedKey MakeFixedKey(const State& s) {
    FixedKey f;
    if (s.rs[kRsLighting]) {
        f.light = 1 | (s.rs[kRsSpecularEnable] ? 2 : 0) | (s.rs[kRsLocalViewer] ? 4 : 0) | (s.rs[kRsNormalizeNormals] ? 8 : 0);
        if (s.rs[kRsColorVertex])
            f.light |= (s.rs[kRsDiffuseMaterialSource] & 3) << 8 | (s.rs[kRsSpecularMaterialSource] & 3) << 10 |
                       (s.rs[kRsAmbientMaterialSource] & 3) << 12 | (s.rs[kRsEmissiveMaterialSource] & 3) << 14;
        for (int i = 0; i < 8; i++)
            if ((s.light_enable >> i) & 1) f.types |= (s.lights[size_t(i)][0] & 3) << (i * 2);
    }
    for (int t = 0; t < 4; t++) {
        uint32_t tci = s.tss[size_t(t)][kTssTexCoordIndex], ttf = s.tss[size_t(t)][kTssTextureTransformFlags];
        uint32_t set = tci & 3, size = 2;
        if (s.vs && (s.vs_handle & 1)) {
            uint32_t ty = s.vs->inputs[9 + set].type;
            size = ty == 0x16 ? 3 : std::min<uint32_t>(ty >> 4, 4);  // 0x16 NORMPACKED3
            if (ty >> 4 > 4) size = 3;
            if (!size) size = 2;
        } else {
            uint32_t code = (s.vs_handle >> (16 + set * 2)) & 3;
            size = code == 0 ? 2 : code == 1 ? 3 : code == 2 ? 4 : 1;
        }
        f.tex[t] = set | ((tci >> 16) & 7) << 4 | (ttf & 7) << 8 | (ttf & 0x100 ? 0x800 : 0) | size << 12;
    }
    return f;
}

// D3D's fixed function vertex pipeline: transform, lighting in view space, texture coordinate generation and transforms
std::string FixedVertexHlsl(uint16_t regs, bool pretransformed, uint64_t decode, const FixedKey& f) {
    std::string s = Common();
    s += kVsHelpers;
    s += VsInputs(regs, decode);
    s += kVsPrologue;
    auto v = [&](int r, const char* def) { return (regs >> r) & 1 ? F("i.v%d", r) : std::string(def); };
    bool lit = !pretransformed && (f.light & 1);
    if (pretransformed) {
        s += "    float4 rhw = " + v(0, "float4(0,0,0,1)") + ";\n";
        s += "    float w = rhw.w == 0 ? 1 : 1 / rhw.w;\n";
        s += "    oPos = float4(rhw.xy, rhw.z / surf.z, w);\n";
    } else {
        s += "    float4 ip = float4(" + v(0, "float4(0,0,0,1)") + ".xyz, 1);\n";
        s += "    float4 cp = mul(ip, wvp);\n";
        s += "    float3 vp = mul(ip, wv).xyz;\n";
        s += "    float3 vn = mul(float4(" + v(2, "float4(0,0,0,0)") + ".xyz, 0), wvn).xyz;\n";
        if (f.light & 8) s += "    vn = dot(vn, vn) > 0 ? normalize(vn) : vn;\n";
    }
    if (lit) {
        // D3DMCS: 1 the vertex diffuse, 2 its specular, else the material
        auto src = [&](int field, const char* mat) {
            uint32_t m = (f.light >> (8 + field * 2)) & 3;
            if (m == 1 && ((regs >> 3) & 1)) return std::string("i.v3");
            if (m == 2 && ((regs >> 4) & 1)) return std::string("i.v4");
            return std::string(mat);
        };
        s += "    float4 md = " + src(0, "mat[0]") + ", ms = " + src(1, "mat[2]") + ", ma = " + src(2, "mat[1]") +
             ", me = " + src(3, "mat[3]") + ";\n";
        s += "    float3 la = 0, ld = 0, ls = 0;\n";
        s += (f.light & 4) ? "    float3 eye = -normalize(vp);\n" : "    float3 eye = float3(0, 0, 1);\n";
        for (int n = 0; n < 8; n++) {
            uint32_t type = (f.types >> (n * 2)) & 3;  // D3DLIGHTTYPE: 1 point, 2 spot, 3 directional
            if (!type) continue;
            int b = n * 7;
            s += "    {\n        float3 l;\n        float att = 1;\n";
            if (type == 3) {
                s += F("        l = -normalize(lt[%d].xyz);\n", b + 4);
            } else {
                s += F("        float3 d = lt[%d].xyz - vp;\n        float dist = length(d);\n        l = d / max(dist, 1e-6);\n", b + 3);
                s += F("        att = dist > lt[%d].w ? 0 : 1 / max(lt[%d].x + lt[%d].y * dist + lt[%d].z * dist * dist, 1e-6);\n", b + 3, b + 5,
                       b + 5, b + 5);
                if (type == 2) {
                    s += F("        float rho = dot(-l, normalize(lt[%d].xyz));\n", b + 4);
                    s += F("        att *= rho > lt[%d].x ? 1 : rho <= lt[%d].y ? 0 : pow(saturate((rho - lt[%d].y) / (lt[%d].x - lt[%d].y)), lt[%d].w);\n",
                           b + 6, b + 6, b + 6, b + 6, b + 6, b + 4);
                }
            }
            s += F("        la += att * lt[%d].rgb;\n", b + 2);
            s += "        float nl = dot(vn, l);\n";
            s += F("        ld += att * lt[%d].rgb * max(nl, 0);\n", b);
            if (f.light & 2) s += F("        if (nl > 0) ls += att * lt[%d].rgb * pow(max(dot(vn, normalize(l + eye)), 0), mat[4].x);\n", b + 1);
            s += "    }\n";
        }
        s += "    oD0 = float4(me.rgb + ma.rgb * (amb.rgb + la) + md.rgb * ld, md.a);\n";
        s += (f.light & 2) ? "    oD1 = float4(ms.rgb * ls, ms.a);\n" : "    oD1 = float4(0,0,0,0);\n";
    } else {
        s += "    oD0 = " + v(3, "float4(1,1,1,1)") + ";\n";
        s += "    oD1 = " + v(4, "float4(0,0,0,0)") + ";\n";
    }
    s += "    oB0 = oD0;\n    oB1 = oD1;\n";
    for (int t = 0; t < 4; t++) {
        uint32_t u = f.tex[t], set = u & 3, gen = (u >> 4) & 7, count = (u >> 8) & 7, size = (u >> 12) & 7;
        std::string in = v(9 + int(set), "float4(0,0,0,1)");
        if (pretransformed || (!gen && !count)) {
            s += F("    oT%d = %s;\n", t, in.c_str());
            continue;
        }
        // generated coordinates are view space; a transform takes the set with 1 after its last element
        std::string c = gen == 1 ? "float4(vn, 1)" : gen == 2 ? "float4(vp, 1)" : gen == 3 ? "float4(reflect(normalize(vp), vn), 1)"
                        : size == 1 ? F("float4(%s.x, 1, 0, 0)", in.c_str()) : size == 2 ? F("float4(%s.xy, 1, 0)", in.c_str())
                        : size == 3 ? F("float4(%s.xyz, 1)", in.c_str()) : in;
        if (!count) {
            s += F("    oT%d = %s;\n", t, c.c_str());
            continue;
        }
        s += F("    {\n        float4 tc = mul(%s, tm[%d]);\n", c.c_str(), t);
        if (u & 0x800) s += F("        oT%d = float4(tc.xy, tc.z, tc[%u]);\n", t, std::min<uint32_t>(count, 4) - 1);
        else s += F("        oT%d = float4(tc.xy, %s, 1);\n", t, count >= 3 ? "tc.z" : "0");
        s += "    }\n";
    }
    s += "    oFog = " + v(5, "float4(0,0,0,0)") + ";\n";
    if (pretransformed) return s + kVsEpilogue;
    s += R"(
    VOut o;
    // like the pretransformed ones, target sized passes end half a console pixel short of the far edges
    float2 sp = (float2(cp.x, -cp.y) / cp.w + 1) * 0.5 * surf.xy;
    if (surf.w == 0 && abs(sp.x - (surf.x - 0.5)) < 1e-3) cp.x = cp.w;
    if (surf.w == 0 && abs(sp.y - (surf.y - 0.5)) < 1e-3) cp.y = -cp.w;
    o.pos = cp;
    o.d0 = saturate(oD0);
    o.d1 = saturate(oD1);
    o.b0 = o.d0;
    o.b1 = o.d1;
    o.tc0 = oT0;
    o.tc1 = oT1;
    o.tc2 = oT2;
    o.tc3 = oT3;
    o.fogamt = fogfactor(abs(cp.z));
    return o;
}
)";
    return s;
}

bool Discards(const PixelKey& k) {
    if (k.alphatest || k.alphakill) return true;
    for (int i = 0; k.programmable && i < 4; i++) {
        if (((k.tex_modes >> (i * 5)) & 0x1F) == 5) return true;  // PS_TEXTUREMODES clipplane
    }
    return false;
}

bool Discards(const State& s) {
    if (s.simple[kMAlphaTestEnable >> 2]) return true;
    for (int i = 0; i < 4; i++) {
        if (s.tss[size_t(i)][kTssAlphaKill] == 4) return true;  // D3DTALPHAKILL_ENABLE
        if (s.ps && ((s.rs[kRsPsTextureModes] >> (i * 5)) & 0x1F) == 5) return true;  // a stage kept whatever is bound
    }
    return false;
}

std::string PixelHlsl(const PixelKey& k) {
    std::string s = Common();
    bool sun = false;
    for (const TexInfo& t : k.tex) sun = sun || (t.flags & kTexSun);
    if (sun) {
        size_t end = s.rfind(kConstantsEnd);
        if (end != std::string::npos) s.replace(end, std::strlen(kConstantsEnd), kConstantsEndSun);
    }
    s += PsHelpers();
    if (sun) s += kPsSun;
    for (int i = 0; i < 4; i++) {
        const TexInfo& t = k.tex[i];
        const char* type = t.shadow ? "Texture2D<float>" : t.kind == kTex3D ? "Texture3D" : t.kind == kTexCube ? "TextureCube" : "Texture2D";
        s += F("%s tex%d : register(t%d);\n%s smp%d : register(s%d);\n", type, i, i, t.shadow ? "SamplerComparisonState" : "SamplerState", i, i);
    }
    s += F("float4 main(VOut i, bool front : SV_IsFrontFace) : SV_Target {\n");
    s += "    float4 tc0 = i.tc0, tc1 = i.tc1, tc2 = i.tc2, tc3 = i.tc3;\n";
    s += k.two_sided ? "    float4 v0 = front ? i.d0 : i.b0, v1 = front ? i.d1 : i.b1;\n" : "    float4 v0 = i.d0, v1 = i.d1;\n";
    s += "    if (misc.w == 0) v1 = float4(0, 0, 0, 1);\n";
    s += "    float4 fogin = float4(fogcolor.rgb, saturate(i.fogamt));\n";
    std::string body;
    if (k.programmable) {
        for (int i = 0; i < 4; i++) {
            body += TexFetch(k, i);
            body += Hooks().Fetched(k.tex[i], i);
        }
        for (int i = 0; i < 4; i++) {
            uint32_t m = (k.tex_modes >> (i * 5)) & 0x1F;
            bool fetch = m && m != 4 && m != 5 && m != 0xA && m != 0x11;
            if (fetch && ((k.alphakill >> i) & 1)) body += F("if (t%d.a == 0) discard;\n", i);
        }
        uint32_t mode0 = k.tex_modes & 0x1F;
        body += F("float4 r0 = float4(0, 0, 0, %s), r1 = 0;\n", mode0 ? "t0.a" : "1");
        Ps ps{k};
        uint32_t n = std::min<uint32_t>(k.count & 0xFF, 8);
        for (uint32_t st = 0; st < n; st++) {
            ps.stage = int(st);
            ps.code += "{\n";
            std::string wc = ps.Stage(k.rgb_in[st], k.rgb_out[st], false);
            std::string wa = ps.Stage(k.a_in[st], k.a_out[st], true);
            ps.code += wc + wa + "}\n";
        }
        body += ps.code;
        if (k.final_abcd || k.final_efg) {
            ps.stage = 8;
            uint32_t fl = k.final_efg & 0xFF;
            ps.clamp_sum = fl & 0x80;
            ps.inv_v1 = fl & 0x40;
            ps.inv_r0 = fl & 0x20;
            ps.e = ps.Input(ParseIn(k.final_efg >> 24), false);
            ps.f = ps.Input(ParseIn((k.final_efg >> 16) & 0xFF), false);
            std::string A = ps.Input(ParseIn(k.final_abcd >> 24), false), B = ps.Input(ParseIn((k.final_abcd >> 16) & 0xFF), false),
                        C = ps.Input(ParseIn((k.final_abcd >> 8) & 0xFF), false), D = ps.Input(ParseIn(k.final_abcd & 0xFF), false),
                        G = ps.Input(ParseIn((k.final_efg >> 8) & 0xFF), true);
            body += "float4 outc;\n";
            body += "outc.rgb = " + D + " + lerp(" + C + ", " + B + ", " + A + ");\n";
            body += "outc.a = " + G + ";\n";
        } else {
            body += "float4 outc = float4(lerp(fogin.rgb, r0.rgb, fogin.a), r0.a);\n";
        }
    } else {
        body += "float4 cur = v0, tmp = 0;\n";
        for (int i = 0; i < 4; i++) {
            const uint32_t* f = k.ff[i];
            if (f[0] <= 1) break;
            const TexInfo& t = k.tex[i];
            std::string uv = TexUv(t, i, (k.proj >> i) & 1 ? F("tc%d.xy / tc%d.w", i, i) : F("tc%d.xy", i));
            std::string cube;
            if (t.flags & kTexSun) {
                std::string light = (t.flags & kTexSunLight) && i > 0 ? F("t%d", i - 1) : std::string("float4(1, 1, 1, 1)");
                body += F("float4 t%d = sunshadow(tex%d, smp%d, tc%d.xyz / tc%d.w, ", i, i, i, i, i) +
                        TexUv(t, i, F("tc%d.xy / tc%d.w", i, i)) + ", " + light + ");\n";
            } else if (t.shadow)
                body += F("float4 t%d = shadow(tex%d, smp%d, ", i, i, i) + TexUv(t, i, F("tc%d.xy / tc%d.w", i, i)) +
                        F(", tc%d.z / tc%d.w, %d, i.pos.xy, texsize[%d].xy);\n", i, i, t.shadow, i);
            else if (t.kind == kTex2D && (t.flags & kTexWide))
                body += F("float4 t%d = spreadtap(tex%d, smp%d, %s, spread[%d], %s);\n", i, i, i, uv.c_str(), i, Hooks().ReadScale(t).c_str());
            else if (t.kind == kTex2D) body += F("float4 t%d = tex%d.Sample(smp%d, %s);\n", i, i, i, uv.c_str());
            else if (t.kind == kTexCube && !(cube = Hooks().Cube(t, i, F("tc%d.xyz", i))).empty()) body += F("float4 t%d = %s;\n", i, cube.c_str());
            else if (t.kind) body += F("float4 t%d = tex%d.Sample(smp%d, tc%d.xyz);\n", i, i, i, i);
            else body += F("float4 t%d = float4(1,1,1,1);\n", i);
            body += Hooks().Fetched(t, i);
            if ((k.alphakill >> i) & 1) body += F("if (t%d.a == 0) discard;\n", i);
            const char* dst = f[8] == 5 ? "tmp" : "cur";
            std::string rgb = FfOp(f[0], FfArg(f[1], i, false), FfArg(f[2], i, false), FfArg(f[3], i, false), i, false);
            std::string a = f[4] <= 1 ? std::string("cur.a") : FfOp(f[4], FfArg(f[5], i, true), FfArg(f[6], i, true), FfArg(f[7], i, true), i, true);
            body += F("{ float3 c3 = saturate(%s); float c1 = saturate(%s); %s = float4(c3, c1); }\n", rgb.c_str(), a.c_str(), dst);
        }
        body += "cur.rgb = saturate(cur.rgb + v1.rgb);\n";
        body += "float4 outc = float4(lerp(fogin.rgb, cur.rgb, fogin.a), cur.a);\n";
    }
    if (k.alphatest) body += "alphatest(outc.a);\n";
    body += Hooks().Output() + "return outc;\n}\n";
    s += body;
    return k.quad ? QuadHlsl(std::move(s)) : s;
}

// the vertex shader hands on the pixel position as drawn and as pulled (e), the pixel shader takes the stretch back out
std::string QuadHlsl(std::string s) {
    auto sub = [&](const char* at, const char* before, const char* after) {
        size_t p = s.find(at);
        if (p != std::string::npos) s.replace(p, std::strlen(at), std::string(before) + at + after);
    };
    sub("    float fogamt : TEXCOORD4;\n", "", "    noperspective float4 e : TEXCOORD5;\n");
    sub("    if (surf.w == 0 && abs(p.x - (surf.x - 0.5)) < 1e-3) p.x = surf.x;\n", "    o.e.xy = p.xy;\n", "");
    sub("    if (surf.w == 0 && abs(p.y - (surf.y - 0.5)) < 1e-3) p.y = surf.y;\n", "", "    o.e.zw = p.xy;\n");
    sub("    o.pos = cp;\n", "    o.e = float4(sp, (float2(cp.x, -cp.y) / cp.w + 1) * 0.5 * surf.xy);\n", "");
    sub("float4 main(VOut i, bool front", R"(float4 unstretch(float4 t, float4 e) {
    float2 d = e.zw - e.xy, g = float2(ddx(e.x), ddy(e.y));
    return t + (d.x ? d.x / g.x * ddx(t) : 0) + (d.y ? d.y / g.y * ddy(t) : 0);
}
)", "");
    for (int i = 0; i < 4; i++) sub(F("i.tc%d", i).c_str(), "unstretch(", ", i.e)");
    return s;
}

namespace {

// A w buffer's reversed depth comes from the clip w alone, zparams.x * (zparams.y / w - 1), exact where z / w had
// already rounded away; a z buffer's is 1 - z. zparams.z is the polygon offset in the units of each
const char* kReversedClipHlsl = R"(float ReversedClip(float z, float w) {
    if (zparams.x != 0) return zparams.x * (zparams.y - (w + zparams.z));
    return w - z - zparams.z * w;
}
)";

bool InsertAfter(std::string& s, const char* at, const char* text) {
    size_t p = s.find(at);
    if (p == std::string::npos) return false;
    s.insert(p + std::strlen(at), text);
    return true;
}

}  // namespace

std::string ReversedDepthHlsl(std::string s) {
    size_t entry = s.find("VOut main(");
    if (entry != std::string::npos) s.insert(entry, kReversedClipHlsl);
    InsertAfter(s, "    o.pos = q;\n", "    o.pos.z = ReversedClip(q.z, q.w);\n");
    InsertAfter(s, "    o.pos = cp;\n", "    o.pos.z = ReversedClip(cp.z, cp.w);\n");
    return s;
}

// a point covers its console pixel, several host pixels on an upscaled target: drawn as a quad of that size
namespace {
// DrawConstants::mirror, from the renderer:
//   [0] this frame's projection: 1 / its x and y scale, its x and y centre      [1] view z = x / (y - depth), then how
//   much of the mirror shows (1 as worked out) and the ripples' clock, seconds counted round 64
//   [2] [3] the same two of the last frame's view                                [4] [5] [6] this frame's view space
//   to the last frame's: x, y and z, each a dot with (position, 1)
//   [7] the surface's up in view space    [8] [9] the world's two level axes: xyz their directions in view space,
//   and with w the dot with (position, 1) is the place along each           [10] the scene's viewport in target pixels
//   [3] z: 1 when the last frame's picture can be read
//   [11] the water's own colour, w 0 for the mirror laid over the title's surface, or how rough a water drawn whole is
//   [12] the sky's colour where the picture has nothing to mirror, w how bright the sun glints
//   [13] the way to the sun in view space
//   [7] w: with it the dot with (position, 1) is the height in the world
//   [14] to [25] what fell in the water of late (d3d::Water::splash): its place in the world, w 0 for none or the
//   seconds since + 16 times its kind (kFell)
const char* kMirror = R"(
Texture2D last : register(t0);
Texture2D<float> lastz : register(t1);
// ripples: direction, 2 pi / length, slope; phase, turns in 64 s
static const float4 kWave[6] = {
    float4(0.9553, 0.2955, 0.02978, 0.09), float4(0.4536, 0.8912, 0.04161, 0.08), float4(-0.4161, 0.9093, 0.06478, 0.08),
    float4(-0.9041, 0.4274, 0.10300, 0.07), float4(-0.6536, -0.7568, 0.16982, 0.06), float4(0.4685, -0.8835, 0.27318, 0.05)};
static const float2 kTurn[6] = {float2(0.3, 21), float2(1.7, 24), float2(4.1, 31), float2(2.2, 39), float2(5.0, 49), float2(0.9, 63)};
// What fell in a water drawn whole, by kind, each measured from the title's own effect for it (none; a rifle's round,
// a heavy gun's, a cannon's; a small thing fallen in, a large one; a grenade, a tank's shell, an artillery shell):
// how far round the place the water is thrown up and churned white (units), how long the white stays (seconds), how
// long its waves run (seconds), and how long its spray takes to fall back (seconds)
static const float4 kFell[9] = {
    float4(1, 1, 1, 1), float4(20, 1.2, 3, 0.5), float4(40, 1.5, 3.5, 0.6), float4(90, 2, 4.5, 1.0), float4(60, 3, 4.5, 0.8),
    float4(120, 3.5, 5.5, 1.0), float4(80, 3.5, 5, 1.2), float4(150, 4.5, 6.5, 1.6), float4(300, 7, 10, 2.5)};
// a water drawn whole: its waves from a long swell to fine chop, most of them with the wind, in the same terms
static const float4 kSea[22] = {
    float4(0.6598, 0.7514, 0.00624, 0.0263), float4(0.9987, 0.0504, 0.00743, 0.0263), float4(0.9251, -0.3797, 0.01102, 0.0375),
    float4(0.8598, 0.5106, 0.01284, 0.0375), float4(0.9940, 0.1095, 0.02061, 0.0450), float4(0.7603, 0.6496, 0.01765, 0.0450),
    float4(0.6642, 0.7475, 0.02884, 0.0525), float4(0.9968, -0.0795, 0.03199, 0.0525), float4(0.9465, -0.3228, 0.05213, 0.0562),
    float4(0.1360, 0.9907, 0.05041, 0.0562), float4(0.8289, 0.5595, 0.09064, 0.0562), float4(0.9999, -0.0152, 0.07921, 0.0562),
    float4(0.9431, 0.3326, 0.15707, 0.0525), float4(0.9959, -0.0910, 0.14370, 0.0525), float4(0.8141, 0.5807, 0.19623, 0.0488),
    float4(0.4682, 0.8836, 0.24692, 0.0488), float4(0.8165, 0.5774, 0.31408, 0.0450), float4(0.6993, 0.7148, 0.40225, 0.0450),
    float4(0.9699, 0.2436, 0.63817, 0.0375), float4(0.9363, 0.3512, 0.49946, 0.0375), float4(0.6894, 0.7244, 0.86192, 0.0338),
    float4(0.9952, -0.0983, 0.91396, 0.0338)};
static const float2 kSeaTurn[22] = {
    float2(4.87, 9), float2(5.49, 10), float2(5.01, 13), float2(1.75, 14), float2(3.17, 17), float2(4.98, 16),
    float2(1.35, 20), float2(0.28, 21), float2(2.93, 27), float2(3.23, 27), float2(0.07, 36), float2(1.26, 34),
    float2(5.21, 48), float2(5.53, 46), float2(4.02, 53), float2(3.40, 60), float2(2.27, 67), float2(2.43, 76),
    float2(5.13, 96), float2(3.71, 85), float2(4.25, 111), float2(1.50, 115)};
// the last frame's pixel a point of its view space falls on
float2 MirrorPixel(float3 q) {
    float2 ndc = q.xy / (q.z * mirror[2].xy) + mirror[2].zw;
    return (ndc * float2(0.5, -0.5) + 0.5) * mirror[10].zw + mirror[10].xy;
}
bool MirrorInside(float2 px) {
    return px.x >= mirror[10].x && px.y >= mirror[10].y && px.x < mirror[10].x + mirror[10].z && px.y < mirror[10].y + mirror[10].w;
}
// the last frame's view z there
float MirrorDepth(float2 px) {
    int2 p = clamp(int2(px), int2(mirror[10].xy), int2(mirror[10].xy + mirror[10].zw) - 1);
    return mirror[3].x / (mirror[3].y - lastz.Load(int3(p, 0)));
}
// where in its view space what it holds there lies, z its view z
float3 MirrorPlace(float2 px, float z) {
    float2 centre = floor(clamp(px, mirror[10].xy, mirror[10].xy + mirror[10].zw - 1)) + 0.5;
    float2 ndc = (centre - mirror[10].xy) / mirror[10].zw * float2(2, -2) + float2(-1, 1);
    return float3((ndc - mirror[2].zw) * mirror[2].xy * z, z);
}
// its colour there, between the four pixels around
float3 MirrorColour(float2 px) {
    float2 hi = mirror[10].xy + mirror[10].zw - 1;
    float2 q = clamp(px - 0.5, mirror[10].xy, hi);
    int2 a = int2(q);
    int2 b = min(a + 1, int2(hi));
    float2 f = q - a;
    float3 c00 = last.Load(int3(a.x, a.y, 0)).rgb;
    float3 c10 = last.Load(int3(b.x, a.y, 0)).rgb;
    float3 c01 = last.Load(int3(a.x, b.y, 0)).rgb;
    float3 c11 = last.Load(int3(b.x, b.y, 0)).rgb;
    return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}
// 0 at the picture's edge, 1 from 8% in: what lies past the edge is not in the picture to be mirrored
float MirrorEdge(float2 px) {
    float2 e = min(px - mirror[10].xy, mirror[10].xy + mirror[10].zw - px) / (0.08 * mirror[10].zw);
    return saturate(min(e.x, e.y));
}
// What the surface at P (up N, view z) mirrors along R, from the last frame's picture: its colour, and in w how
// much of it the picture gives (0 where it holds nothing for that line).
float4 MirrorFind(float3 P, float3 R, float3 N, float z) {
    // along it through the last frame's picture: steps that grow, the first of them that lies just behind what the
    // picture holds there is what the surface mirrors. Not the surface itself: seen flat, a pixel of it spans many
    // units of depth and a step along the line is soon behind the one depth the picture has for it. What lies in the
    // surface's own level (within a hundredth of the way there, for a sheet that is not quite flat) is passed over.
    float3 Pl = float3(dot(P, mirror[4].xyz) + mirror[4].w, dot(P, mirror[5].xyz) + mirror[5].w, dot(P, mirror[6].xyz) + mirror[6].w);
    float3 Rl = float3(dot(R, mirror[4].xyz), dot(R, mirror[5].xyz), dot(R, mirror[6].xyz));
    float3 Nl = float3(dot(N, mirror[4].xyz), dot(N, mirror[5].xyz), dot(N, mirror[6].xyz));
    float reach = max(z * 2, 400);
    float3 col = float3(0, 0, 0);
    float show = 0;
    float before = 0;
    bool hit = false;
    for (int s = 1; s <= 32; s++) {
        float f = s / 32.0;
        float t = reach * f * f;
        float3 q = Pl + Rl * t;
        float2 px = MirrorPixel(float3(q.xy, max(q.z, 1)));
        if (!hit && q.z > 1 && MirrorInside(px)) {
            float d = q.z - MirrorDepth(px);
            float3 off = MirrorPlace(px, q.z - d) - Pl;
            if (d > 0 && d < max((t - before) * 1.5, q.z * 0.03) && abs(dot(off, Nl)) > 1 + 0.01 * length(off)) {
                hit = true;
                col = MirrorColour(px);
                show = MirrorEdge(px);
            }
        }
        before = t;
    }
    // none: what lies far off the way the line goes (the sky, a far shore), where nothing nearer stands before it
    if (!hit && Rl.z > 1e-4) {
        float2 px = MirrorPixel(Rl);
        if (MirrorInside(px) && MirrorDepth(px) > Pl.z + reach) {
            col = MirrorColour(px);
            show = MirrorEdge(px);
        }
    }
    return float4(col, show);
}
// A water drawn whole: how near something rises through the surface round P (a shore, a hull, a post), 1 at it to 0
// from kFoam units out. Points of the surface round P, in three rings: what the last frame's picture holds over each
// is such a thing where it lies a little above the surface's level (a beach rises a unit in a ring's width; no more
// than the ring is wide: a bridge over the water is not in it) and near the point itself (what stands before the
// water, nearer the eye, is not).
static const float kFoam = 64;
float MirrorShore(float3 P, float3 N) {
    float3 Pl = float3(dot(P, mirror[4].xyz) + mirror[4].w, dot(P, mirror[5].xyz) + mirror[5].w, dot(P, mirror[6].xyz) + mirror[6].w);
    float3 Nl = float3(dot(N, mirror[4].xyz), dot(N, mirror[5].xyz), dot(N, mirror[6].xyz));
    float3 Xl = float3(dot(mirror[8].xyz, mirror[4].xyz), dot(mirror[8].xyz, mirror[5].xyz), dot(mirror[8].xyz, mirror[6].xyz));
    float3 Yl = float3(dot(mirror[9].xyz, mirror[4].xyz), dot(mirror[9].xyz, mirror[5].xyz), dot(mirror[9].xyz, mirror[6].xyz));
    float near = 0;
    for (int r = 1; r <= 3; r++) {
        float d = r * (kFoam / 3.0);
        for (int k = 0; k < 6; k++) {
            float turn = k * 1.0471976 + r * 0.35;
            float3 q = Pl + (Xl * cos(turn) + Yl * sin(turn)) * d;
            float2 px = MirrorPixel(float3(q.xy, max(q.z, 1)));
            if (q.z > 1 && MirrorInside(px)) {
                float3 held = MirrorPlace(px, MirrorDepth(px));
                float over = dot(held - Pl, Nl);
                if (over > 0.4 + 0.003 * q.z && over < d + 4 && length(held - q) < 1.5 * d + 4 + 0.005 * q.z) {
                    near = max(near, (4 - r) / 3.0);
                }
            }
        }
    }
    return near;
}
float4 main(VOut i) : SV_Target {
    float4 vp = mirror[10];
    float2 ndc = (i.pos.xy - vp.xy) / vp.zw * float2(2, -2) + float2(-1, 1);
    float z = mirror[1].x / (mirror[1].y - i.pos.z);
    float3 P = float3((ndc - mirror[0].zw) * mirror[0].xy * z, z);
    float3 V = normalize(P);
    float3 N = mirror[7].xyz;
    // ripples over the level plane, each fading where it is finer than the pixels that show it
    float2 place = float2(dot(P, mirror[8].xyz) + mirror[8].w, dot(P, mirror[9].xyz) + mirror[9].w);
    float foot = z * mirror[0].x * 2 / vp.z / max(abs(dot(V, N)), 0.02);
    float2 slope = float2(0, 0);
    float sea = mirror[11].w;
    float crest = 0;
    float churn = 0;
    if (sea > 0) {
        for (int j = 0; j < 22; j++) {
            float len = 6.2831853 / kSea[j].z;
            float a = kSea[j].w * sea * saturate(len / (foot * 3) - 0.5);
            float arg = dot(place, kSea[j].xy) * kSea[j].z + kSeaTurn[j].x - mirror[1].w * kSeaTurn[j].y * 0.09817477;
            slope += kSea[j].xy * (cos(arg) * a);
            crest += sin(arg) * a;
        }
        // What fell in it (the title's own splash stands over this, drawn after). Round each place, by its kind's
        // measure: a hollow where it went in, its rim running out in the first moments; the water churned white
        // out to that measure, breaking up into patches and clearing as it ages; the spray falling back as specks
        // over a wider round; and a packet of waves as long as the splash is large, its front spreading and
        // widening, a line of foam on its first crest. Only on a water at the height of the place.
        float level = dot(P, mirror[7].xyz) + mirror[7].w;
        for (int e = 0; e < 12; e++) {
            float4 fell = mirror[14 + e];
            [branch] if (fell.w > 0 && abs(level - fell.z) < 40) {
                int kind = int(min(fell.w / 16, 8.5));
                float age = fell.w - kind * 16;
                float4 size = kFell[kind];
                float wide = size.x;
                float2 off = place - fell.xy;
                float r = length(off);
                float2 out_way = off / max(r, 0.001);
                // the churned water
                float u = saturate(age / size.y);
                float grown = wide * (1 - exp(-age * 6 / size.y)) * (1 + 0.35 * u);
                float2 q = off * (9 / wide);
                float lace = 0.5 + 0.5 * sin(q.x + 1.7 * sin(q.y * 1.3 + fell.x)) * sin(q.y * 1.1 + 1.9 * sin(q.x * 0.9 + fell.y));
                float cover = saturate((grown * (0.8 + 0.4 * lace) - r) / (wide * 0.22 + 1));
                float white = saturate(cover * (1.5 - 1.1 * u) - lace * (0.2 + 1.2 * u)) * step(age, size.y);
                // the waves, from its edge
                float len = max(wide * 0.45, 14);
                float front = grown * 0.7 + (10 * sqrt(len) + 20) * age;
                float fade = saturate(1 - age / size.z) * saturate(age * 3);
                float fine = saturate(len / (foot * 3) - 0.5);
                float x = (r - front) / (len * 2 + front * 0.2);
                float env = exp(-x * x) * fade * fine / sqrt(1 + r / (len * 6));
                // (not the same all the way round)
                env *= 0.7 + 0.3 * sin(dot(out_way, float2(3.1, 1.7)) * 2.3 + fell.x);
                float steep = min(0.3 + wide / 350, 0.8);
                float turn = (r - front) * 6.2831853 / len;
                slope += out_way * (cos(turn) * env * steep);
                crest += sin(turn) * env * steep * 0.5;
                float rim = (r - front - len * 0.25) / (len * 0.2);
                white = max(white, exp(-rim * rim) * fade * fade * fine * (0.25 + 0.6 * lace));
                // the hollow
                float y = r / (wide * (0.25 + 0.75 * saturate(age * 4)));
                slope += out_way * (y * exp(-y * y) * saturate(1 - age / (0.5 + wide / 200)) * 1.6 * saturate(wide / (foot * 4) - 0.3));
                // the spray falling back
                float falls = saturate(age / 0.3) * saturate(2 - age / size.w) * saturate(1.6 - r / wide);
                float cell = max(wide / 16, 3);
                float2 box = floor(off / cell);
                float h = frac(sin(dot(box, float2(12.9898, 78.233)) + floor(age * 12) * 37.719) * 43758.5453);
                float2 drop = off / cell - box - 0.5 - (float2(frac(h * 57.31), frac(h * 113.7)) - 0.5) * 0.5;
                float sharp = saturate(cell / foot * 0.25 - 0.5);
                float speck = h > 1 - 0.35 * falls ? saturate((0.25 - length(drop)) * 8) : 0.0;
                white = max(white, speck * 0.85 * sharp + falls * 0.12 * (1 - sharp));
                churn = max(churn, white);
            }
        }
    } else {
        for (int k = 0; k < 6; k++) {
            float len = 6.2831853 / kWave[k].z;
            float a = kWave[k].w * saturate(len / (foot * 3) - 0.5);
            slope += kWave[k].xy * (cos(dot(place, kWave[k].xy) * kWave[k].z + kTurn[k].x - mirror[1].w * kTurn[k].y * 0.09817477) * a);
        }
    }
    float3 n = normalize(N - slope.x * mirror[8].xyz - slope.y * mirror[9].xyz);
    float3 R = reflect(V, n);
    // a line that would go under the surface is laid along it
    float rise = dot(R, N);
    if (rise < 0.02) R = normalize(R - (rise - 0.02) * N);
    float4 found = float4(0, 0, 0, 0);
    if (mirror[3].z > 0.5) found = MirrorFind(P, R, N, z);
    // more of the mirror the flatter the surface is seen
    float facing = saturate(-dot(V, n));
    float fresnel = 0.02 + 0.98 * pow(1 - facing, 5);
    // laid over the title's surface: never all of it, and none of it in the fog
    if (sea <= 0) return float4(found.rgb, min(fresnel, 0.6) * mirror[1].z * found.w * saturate(i.fogamt));
    // The water whole. Looked down into it is its own colour darkened, a little to the green and blue; a crest that
    // faces the eye lets some light back through; the rest is the mirror, the sky's colour where the picture has
    // nothing for it.
    float3 tone = mirror[11].rgb * saturate(i.d0.rgb);
    float3 deep = tone * float3(0.42, 0.50, 0.52);
    float3 scatter = tone * float3(0.75, 0.95, 0.90);
    float light = saturate(crest * 2.2 + 0.15) * saturate(1 - facing * 1.5);
    float3 body = deep + (scatter - deep) * (light * 0.5);
    float3 seen = lerp(mirror[12].rgb, found.rgb, found.w);
    float3 c = body * (1 - fresnel) + seen * min(fresnel, 0.85);
    // the sun's glint, where the mirrored line goes to it
    c += float3(1.0, 0.97, 0.9) * (pow(saturate(dot(R, mirror[13].xyz)), 260) * mirror[12].w);
    // white where what fell in churned it; and foam where a shore or a hull rises through it: whole at the edge, in
    // patches by the waves further out, none where the rings round a pixel are finer than the pixels (and none where
    // no shore is near: the waves' crests alone make no foam)
    float foam = churn;
    if (mirror[3].z > 0.5) {
        float shore = MirrorShore(P, N) * saturate(2 - foot * 6 / kFoam);
        foam = max(foam, saturate(shore * 1.3 - 0.45 + crest * 2.5) * saturate(shore * 6));
    }
    c = lerp(c, saturate(mirror[12].rgb * 1.5 + 0.1), foam * 0.9);
    // far off, where its waves are finer than the pixels, the title's own surface is left to show; and in the fog
    return float4(c, (1 - saturate((z - 5000) / 9000)) * saturate(i.fogamt));
}
)";
}  // namespace

std::string MirrorHlsl() {
    std::string s = Common();
    size_t end = s.rfind(kConstantsEnd);
    if (end != std::string::npos) {
        s.replace(end, std::strlen(kConstantsEnd), "    float4 zparams;\n    float4 sun[5];\n    float4 mirror[26];\n};\n");
    }
    return s + kMirror;
}

std::string PointHlsl() {
    return Common() + R"(
[maxvertexcount(4)]
void main(point VOut v[1], inout TriangleStream<VOut> strip) {
    float2 half_pixel = v[0].pos.w / surf.xy;  // in clip units, the target spans 2 across surf.xy pixels
    // a target is read half a host pixel back (surf.w host pixels per console pixel): left and top reach that far
    float2 back = (0.5 - 1e-3) * 2 * half_pixel / surf.w;
    for (int i = 0; i < 4; i++) {
        VOut o = v[0];
        o.pos.x += (i & 1) ? half_pixel.x : -half_pixel.x - back.x;
        o.pos.y += (i & 2) ? -half_pixel.y : half_pixel.y + back.y;
        strip.Append(o);
    }
}
)";
}

}  // namespace gpu

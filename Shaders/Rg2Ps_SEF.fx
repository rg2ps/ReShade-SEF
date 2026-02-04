/*
   Real Time Simulated Exposure Fusion Technique
   Based on the: "Simulated Exposure Fusion. Charles Hessel (2019): https://www.ipol.im/pub/art/2019/279/"

   Written for ReShade by RG2PS (c) 2026. Apache 2.0 License.
   Any file parts redistribution are governed by the current license agreement.
*/

uniform float _Alpha
<
    ui_label = "Fusion Range";
    ui_type = "slider";
    ui_min = 6.0; ui_max = 32.0;
> = 16.0;

uniform float _Range
<
    ui_label = "Fusion Balance";
    ui_type = "slider";
    ui_min = 0.45; ui_max = 0.8;
> = 0.8;

uniform float _Beta
<
    ui_label = "Compression Range";
    ui_type = "slider";
    ui_min = 0.4; ui_max = 0.8;
> = 0.5;

uniform float _MidGrey
<
    ui_label = "Scene Midgrey";
    ui_type = "drag";
    ui_min = 0.160; ui_max = 0.200;
> = 0.180;

uniform float _Gamma
<
    ui_label = "Fusion Pre-Gamma";
    ui_type = "drag";
    ui_min = 0.0; ui_max = 1.0;
> = 0.5;

uniform bool _Debug
<
    ui_label = "Visualize Exposures Map";
    ui_type = "radio";
> = false;

/*=============================================================================
/   Buffer Samplers Definition
/============================================================================*/
#ifndef MAX_OF_EXPOSURES
    #define MAX_OF_EXPOSURES 16
#endif

#ifndef DITHER_BIT_DEPTH
    #define DITHER_BIT_DEPTH 8
#endif

#include "ReShade.fxh"

texture2D texLowPass_0	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = R8; };
texture2D texLowPass_1	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = R8; };
texture2D texLowPass_2	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = R8; };
texture2D texLowPass_3	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = R8; };
texture2D texLowPass_4	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = R8; };
texture2D texLowPass_5	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = R8; };
texture2D texLowPass_6	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = R8; };
sampler sLowPass_0		{ Texture = texLowPass_0; };
sampler sLowPass_1		{ Texture = texLowPass_1; };
sampler sLowPass_2		{ Texture = texLowPass_2; };
sampler sLowPass_3		{ Texture = texLowPass_3; };
sampler sLowPass_4		{ Texture = texLowPass_4; };
sampler sLowPass_5		{ Texture = texLowPass_5; };
sampler sLowPass_6		{ Texture = texLowPass_6; };

texture2D texInChannel
{ 
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = R8; 
};

sampler sInChannel
{ 
    Texture = texInChannel; 
};

texture2D texPyramidResult
{ 
    Width = BUFFER_WIDTH;   
    Height = BUFFER_HEIGHT;  
    Format = R8; 
};

sampler sPyramidResult
{ 
    Texture = texPyramidResult; 
};

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float3 from_hdr(float3 x) 
{ 
    return x * rsqrt(1.0 + x * x);
} 

float3 to_hdr(float3 x) 
{
    return x * rsqrt(1.0 - x * x + (1.0 / 255.0));
}

float safe_sqrt(float x)
{
    return sqrt(abs(x)) * sign(x);
}

float safe_pow2(float x)
{
    return x * x * sign(x);
}

float weyl(float2 p)
{
    return frac(0.5 + p.x * 0.7548776662467 + p.y * 0.569840290998);
}

/*=============================================================================
/   Shader Entry Points
/============================================================================*/
void channel(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb;

    output = (color.r + color.g + color.b) / 3.0;

    if (output < 0.003921568627) output += 0.0625;
}

float downsample(sampler2D s, float2 uv, float L)
{
    float2 xy = BUFFER_PIXEL_SIZE * exp2(L);

    float a = tex2Dlod(s, float4(uv.x - xy.x, uv.y + xy.y, 0, 0));
    float b = tex2Dlod(s, float4(uv.x,        uv.y + xy.y, 0, 0));
    float c = tex2Dlod(s, float4(uv.x + xy.x, uv.y + xy.y, 0, 0));

    float d = tex2Dlod(s, float4(uv.x - xy.x, uv.y, 0, 0));
    float e = tex2Dlod(s, float4(uv.x,        uv.y, 0, 0));
    float f = tex2Dlod(s, float4(uv.x + xy.x, uv.y, 0, 0));

    float g = tex2Dlod(s, float4(uv.x - xy.x, uv.y - xy.y, 0, 0));
    float h = tex2Dlod(s, float4(uv.x,        uv.y - xy.y, 0, 0));
    float i = tex2Dlod(s, float4(uv.x + xy.x, uv.y - xy.y, 0, 0));

    float window = e * 4.0;
    window += (b + d + f + h) * 2.0;
    window += (a + c + g + i);

    return window / 16.0;
}

void dl_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sInChannel, texcoord, 0);
}

void dl_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sLowPass_0, texcoord, 1);
}

void dl_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sLowPass_1, texcoord, 2);
}

void dl_3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sLowPass_2, texcoord,  3);
}

void dl_4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sLowPass_3, texcoord, 4);
}

void dl_5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sLowPass_4, texcoord, 5);
}

void dl_6(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sLowPass_5, texcoord, 6);
}

void pyramid(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{   
    output = 
    (
        tex2D(sLowPass_0, texcoord) + 
        tex2D(sLowPass_1, texcoord) + 
        tex2D(sLowPass_2, texcoord) + 
        tex2D(sLowPass_3, texcoord) + 
        tex2D(sLowPass_4, texcoord) + 
        tex2D(sLowPass_5, texcoord) + 
        tex2D(sLowPass_6, texcoord)) * 0.14285714;
}

/*=============================================================================
/   Main Shader Workflow
/============================================================================*/
float remap_lowest(float t, int k, int N_max) 
{
    // f∗(t,k) = α^|k|/N_max(t − 1) + 1 (for k < 0)
    float lambda = pow(sqrt(_Alpha), abs(k) / float(N_max));
    return lambda * (t - 1.0) + 1.0;
}

float remap_higher(float t, int k, int N_max) 
{
    // f(t,k) = α^k/N_max(t) (for k >= 0)
    float lambda = pow(sqrt(_Alpha), k / float(N_max));
    return lambda * t;
}

float clip_exposure(float t, int k, int N_star, int N, float beta) 
{
    // eq. 4: ρ(k) = 1 - β/2 - (k+N)(1-β)/(N+N)
    float rho = 1.0 - beta/2.0 - (k + N_star) * (1.0 - beta) / (N + N_star);
    float a = beta / 2.0 + 0.125;
    float b = beta / 2.0 - 0.125;

    if (abs(t - rho) <= beta / 2.0) {
        return t;
    } else {
        return sign(t - rho) * (0.125 / (abs(t - rho) - b)) + rho;
    }
}

float solve_exposure(float t, int k, int N_max, int N_star, int N, float beta) 
{
    float remapped;
    if (k < 0) {
        remapped = remap_lowest(t, k, N_max);
    } else {
        remapped = remap_higher(t, k, N_max);
    }
    return clip_exposure(remapped, k, N_star, N, beta);
}

float find_best_exposure(float t) 
{
    // eq.8: k(x) = exp(-(b_u_k(x) - 0.5)²/2σ²)
    return exp(-(t - 0.5) * (t - 0.5) / 0.08); 
}

float contrast_remap(float t, int k, int N_star, int N, float beta) 
{
    // eq.10 : g'(t,k) = λ²/(|t-ρ(k)|-b)²
    float rho = 1.0 - beta/2.0 - (k + N_star) * (1.0 - beta) / (N + N_star);
    if (abs(t - rho) <= beta / 2.0) {
        return 1.0;
    } else {
        float b = beta / 2.0 - 0.125;
        return 0.003921568627 / ((abs(t - rho) - b) * (abs(t - rho) - b));
    }
}

float find_best_contrast(float t, int k, int N_max, int N_star, int N, float beta) 
{
    // eq.9
    float lambda_k;
    if (k < 0) {
        lambda_k = pow(sqrt(_Alpha), abs(k) / float(N_max));
        return lambda_k * contrast_remap(t, k, N_star, N, beta);
    } else {
        lambda_k = pow(sqrt(_Alpha), k / float(N_max));
        return lambda_k * contrast_remap(t, k, N_star, N, beta);
    }
}

float get_fusion_weights(float t, int k, int N_max, int N_star, int N, float beta) 
{
    // eq. 11
    float w_e = find_best_exposure(t);
    float w_c = find_best_contrast(t, k, N_max, N_star, N, beta);
    return w_e * w_c;
}

float midgrey_value(float x)
{
    return exp2(log2(_MidGrey / 0.18) * 4.0 * x); // 8 stops
}

// the original paper proposes using local laplacian for each exposure, what very expensive in real time. 
// instead that I use the separate extremes/moments processing via weighted mean: v = I * √[(Σ(E²*w)/Σw) / (Σ(√E*w)/Σw)]
void fusion_integral(inout float3 color, float L, int N_max, int N_star, int N, float beta, float r) 
{
    float2 sum = 0.0;
    float2 total = 0.0;

    float top = solve_exposure(L, -N_star, N_max, N_star, N, beta);
    float bottom = solve_exposure(L, 0, N_max, N_star, N, beta);

    [loop]
    for (int k = -N_star; k <= N; k++) 
    {
        float t = frac(r + float(k)) / float(N_star);
        float exposure = solve_exposure(L, k, N_max, N_star, N, beta);
        float weight = get_fusion_weights(exposure, k, N_max, N_star, N, saturate(beta - t));

        // process lights
        [flatten]
        if (k < 0) 
        {
            float current_exposure = safe_pow2(exposure);

            top = current_exposure;

            sum.x += top * weight;
            total.x += weight;
        }

        // process darks
        [flatten]
        if (k >= 0) 
        {
            float current_exposure = safe_sqrt(exposure);

            bottom = current_exposure;
            
            sum.y += bottom * weight;
            total.y += weight;
        }
    }
    
    float2 moments = sum / (total + 1e-6);
    moments += 0.015625; // some bias to avoid zero dev

	float3 median = _Debug ? 0.5 : color;
    float3 m_clip = median * saturate(moments.x * midgrey_value(+1));
    float3 m_lift = median / saturate(moments.y * midgrey_value(-1));
    
    color = min(65535.0, sqrt(m_clip * m_lift));
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target0)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, vpos, 0).rgb;
    float L = tex2Dlod(sPyramidResult, float4(texcoord, 0, 0));

    L = lerp(L*L, L*L*L, _Gamma); // to linear (or cubic)

    float random = weyl(vpos.xy);
    
    const int M = MAX_OF_EXPOSURES;
    int N_star = (int)round(float(M - 1) * _Range);
    int N = (M - 1) - N_star;
    int N_max = max(N_star, N);
    
    color = to_hdr(color);
    
    fusion_integral(color, L, N_max, N_star, N, _Beta, random);

    color = from_hdr(color);

    float bit_depth = exp2(DITHER_BIT_DEPTH) - 1;

    float3 qu_min = floor(color * bit_depth) / bit_depth;
    float3 qu_max = ceil(color * bit_depth) / bit_depth;

    float3 error = saturate((color - qu_min) / (qu_max - qu_min));

    color = lerp(qu_min, qu_max, step(random, error));
    
    output = saturate(color);
}

/*=============================================================================
/   Technique Definition
/============================================================================*/
technique Rg2Ps_SEF < 
ui_label = "Simulated Exposure Fusion";
ui_tooltip = "									Simulated Exposure Fusion \n\n" "___________________________________________________________________________________________________\n\n" "SEF is state of art offline local image processing technique that allows exposure adjustment based\n" "on regional image content. This implementation is one of the few that can works in real time.\n\n" " - Developed by RG2PS - "; >
{
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = channel;
	    RenderTarget = texInChannel;
    }

    #define process(i) pass { VertexShader = PostProcessVS; PixelShader = dl_##i; RenderTarget = texLowPass_##i; }

    process(0)
    process(1)
    process(2)
    process(3)
    process(4)
    process(5)
    process(6)

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = pyramid;
	    RenderTarget = texPyramidResult;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}

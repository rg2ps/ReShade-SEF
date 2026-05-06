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
    ui_min = 8.0; ui_max = 32.0;
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
    ui_min = 0.2; ui_max = 0.8;
> = 0.5;

uniform float _MidGrey
<
    ui_label = "Midgrey";
    ui_type = "slider";
    ui_min = 0.160; ui_max = 0.200;
> = 0.180;

uniform bool _Debug
<
    ui_label = "Visualize Exposures";
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

texture2D texIMG_s0	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = R8; };
texture2D texIMG_s1	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = R8; };
texture2D texIMG_s2	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = R8; };
texture2D texIMG_s3	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = R8; };
texture2D texIMG_s4	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = R8; };
texture2D texIMG_s5	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = R8; };
texture2D texIMG_s6	{ Width = BUFFER_WIDTH >> 6; Height = BUFFER_HEIGHT >> 6; Format = R8; };
sampler sIMG_s0		{ Texture = texIMG_s0; };
sampler sIMG_s1		{ Texture = texIMG_s1; };
sampler sIMG_s2		{ Texture = texIMG_s2; };
sampler sIMG_s3		{ Texture = texIMG_s3; };
sampler sIMG_s4		{ Texture = texIMG_s4; };
sampler sIMG_s5		{ Texture = texIMG_s5; };
sampler sIMG_s6		{ Texture = texIMG_s6; };

texture2D texHDRBuffer
{ 
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = R8; 
};

sampler sHDRBuffer
{ 
    Texture = texHDRBuffer; 
};

texture2D texGaussianIMG
{ 
    Width = BUFFER_WIDTH;   
    Height = BUFFER_HEIGHT;  
    Format = R8; 
};

sampler sGaussianIMG
{ 
    Texture = texGaussianIMG; 
};

texture2D texCollapsedExposures
{ 
    Width = BUFFER_WIDTH;   
    Height = BUFFER_HEIGHT;  
    Format = RG16F; 
};

sampler sCollapsedExposures
{ 
    Texture = texCollapsedExposures; 
};

/*=============================================================================
/   Global Helper Functions
/============================================================================*/
float luminance(float3 x)
{
    return dot(x, float3(0.2126729, 0.7151522, 0.072175));
}

float3 from_hdr(float3 x) 
{ 
    return x * rsqrt(1.0 + x * x);
} 

float3 to_hdr(float3 x) 
{
    return x * rsqrt(1.0 - x * x + (1.0 / 255.0));
}

float safesqrt(float x)
{
    return sqrt(abs(x)) * sign(x);
}

float safepow2(float x)
{
    return x * x * sign(x);
}

float goldenratio_IGN(float2 p)
{
    return frac(0.5 + p.x * 0.7548776662467 + p.y * 0.569840290998);
}

/*=============================================================================
/   Shader Entry Points
/============================================================================*/
void hdrimg(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float color : SV_Target)
{
    color = luminance(max(1e-3, tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb));
}

float downsample(sampler2D s, float2 uv, float mip)
{
    float2 xy = BUFFER_PIXEL_SIZE * exp2(mip);

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

float2 samplecross(sampler2D s, float2 uv)
{
    float2 tap = BUFFER_PIXEL_SIZE * 1.5;
    return 
        0.25 * tex2Dlod(s, float4(uv + float2(-tap.x, 0), 0, 0)).xy +
        0.25 * tex2Dlod(s, float4(uv + float2( tap.x, 0), 0, 0)).xy +
        0.25 * tex2Dlod(s, float4(uv + float2(0, -tap.y), 0, 0)).xy +
        0.25 * tex2Dlod(s, float4(uv + float2(0,  tap.y), 0, 0)).xy;
}

void s0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sHDRBuffer, texcoord, 0);
}

void s1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sIMG_s0, texcoord, 1);
}

void s2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sIMG_s1, texcoord, 2);
}

void s3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sIMG_s2, texcoord,  3);
}

void s4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sIMG_s3, texcoord, 4);
}

void s5(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sIMG_s4, texcoord, 5);
}

void s6(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sIMG_s5, texcoord, 6);
}

void gaussianmap(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{   
    output = 
	(
		tex2D(sIMG_s0, texcoord) + 
		tex2D(sIMG_s1, texcoord) + 
		tex2D(sIMG_s2, texcoord) + 
		tex2D(sIMG_s3, texcoord) + 
		tex2D(sIMG_s4, texcoord) + 
		tex2D(sIMG_s5, texcoord) + 
		tex2D(sIMG_s6, texcoord)
	) * 0.14285714;
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

float keyEV(float x, float K = 8)
{
    return exp2(0.5 * log2(_MidGrey / 0.18) * K * x);
}

// The original paper proposes using local laplacian for each exposure, what very expensive in real time. 
// Instead that I use the separate extremes/moments processing via weighted mean: v = I * √[(Σ(E²*w)/Σw) / (Σ(√E*w)/Σw)]
// To ensure continuous that each pixel receives a number of exposures proportional to the maximum number of generated exposures per frame, 
// we use Monte Carlo integration. This ensures such convergence that each pixel ultimately receives a random exposure value 
// from 1 to the maximum number per frame (M)
float3 fusionmap(float3 x, float2 ev)
{
    return sqrt((x * ev.x) * (x / ev.y));
}

float2 fusion_integral(float map, int N_max, int N_star, int N, float beta, float2 vpos) 
{
    float2 csum = 0.0;
    float2 wsum = 0.0;

    float2 key = float2(keyEV(+1), keyEV(-1));

    float top = solve_exposure(map, -N_star, N_max, N_star, N, beta);
    float bottom = solve_exposure(map, 0, N_max, N_star, N, beta);

    [loop]
    for (int k = -N_star; k <= N; k++) 
    {
        float running = frac(goldenratio_IGN(vpos.xy) + float(k)) / float(N_star);
        float exposure = solve_exposure(map, k, N_max, N_star, N, beta - running);
        float weight = get_fusion_weights(exposure, k, N_max, N_star, N, beta + running);

        // process lights
        [flatten]
        if (k < 0) {
            float current_exposure = safepow2(exposure);

            top = current_exposure;

            csum.x += top * weight;
            wsum.x += weight;
        }

        // process darks
        [flatten]
        if (k >= 0) {
            float current_exposure = safesqrt(exposure);

            bottom = current_exposure;
            
            csum.y += bottom * weight;
            wsum.y += weight;
        }
    }
    
    return (csum / wsum) * key;
}

void collapsedEV(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float2 output : SV_Target)
{
    float map = tex2Dlod(sGaussianIMG, float4(texcoord, 0, 0));

    const int M = MAX_OF_EXPOSURES;

    int N_star = (int)round(float(M - 1) * _Range);
    int N = (M - 1) - N_star;
    int N_max = max(N_star, N);

    output = fusion_integral(pow(map, 2.2), N_max, N_star, N, _Beta, vpos.xy);
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target0)
{
    float3 color = tex2Dfetch(ReShade::BackBuffer, vpos, 0).rgb;
    float2 range = samplecross(sCollapsedExposures, texcoord);

    color = to_hdr(color);
    color = fusionmap(_Debug ? 0.5 : color, range);
    color = from_hdr(color);

    float bit_depth = exp2(DITHER_BIT_DEPTH) - 1;
    float3 qu_min = floor(color * bit_depth) / bit_depth;
    float3 qu_max = ceil(color * bit_depth) / bit_depth;
    float3 error = saturate((color - qu_min) / (qu_max - qu_min));

    color = lerp(qu_min, qu_max, step(goldenratio_IGN(vpos.xy), error));
    
    output = color;
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
	    PixelShader = hdrimg;
	    RenderTarget = texHDRBuffer;
    }

    #define process(i) pass { VertexShader = PostProcessVS; PixelShader = s##i; RenderTarget = texIMG_s##i; }

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
	    PixelShader = gaussianmap;
	    RenderTarget = texGaussianIMG;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = collapsedEV;
	    RenderTarget = texCollapsedExposures;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}
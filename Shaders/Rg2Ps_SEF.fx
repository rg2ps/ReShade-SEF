/*
   Real Time Simulated Exposure Fusion Technique for ReShade
   Writen for ReShade by RG2PS (c) 2025 Apache 2.0 License (see LICENSE)
   Based on the: "Simulated Exposure Fusion. Charles Hessel (2019): https://www.ipol.im/pub/art/2019/279/"
*/

uniform float alpha
<
    ui_label = "Fusion Range";
    ui_tooltip = "The covered range exposures";
    ui_type = "slider";
    ui_min = 2.0; ui_max = 16.0;
> = 8.0;

uniform float re
<
    ui_label = "Fusion Balance";
    ui_type = "slider";
    ui_min = 0.125; ui_max = 1.0;
> = 0.5;

uniform float zone
<
    ui_label = "Metering Zone";
    ui_tooltip = "Influence region of the effect";
    ui_type = "slider";
    ui_min = 0.2; ui_max = 0.8;
> = 0.5;

uniform float midgrey
<
    ui_label = "Midgrey Value";
    ui_tooltip = "0.18 is neutral";
    ui_type = "drag";
    ui_min = 0.01; ui_max = 0.5;
> = 0.16;

uniform bool debug
<
    ui_label = "Enable Debug Mode";
    ui_type = "radio";
> = false;

/*=============================================================================
=============================================================================*/
#include "ReShade.fxh"
texture2D texGL_0	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = R8; };
texture2D texGL_1	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = R8; };
texture2D texGL_2	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = R8; };
texture2D texGL_3	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = R8; };
texture2D texGL_4	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = R8; };
texture2D texGS_0	{ Width = BUFFER_WIDTH >> 1; Height = BUFFER_HEIGHT >> 1; Format = R8; };
texture2D texGS_1	{ Width = BUFFER_WIDTH >> 2; Height = BUFFER_HEIGHT >> 2; Format = R8; };
texture2D texGS_2	{ Width = BUFFER_WIDTH >> 3; Height = BUFFER_HEIGHT >> 3; Format = R8; };
texture2D texGS_3	{ Width = BUFFER_WIDTH >> 4; Height = BUFFER_HEIGHT >> 4; Format = R8; };
texture2D texGS_4	{ Width = BUFFER_WIDTH >> 5; Height = BUFFER_HEIGHT >> 5; Format = R8; };
sampler sGL_0		{ Texture = texGL_0; };
sampler sGL_1		{ Texture = texGL_1; };
sampler sGL_2		{ Texture = texGL_2; };
sampler sGL_3		{ Texture = texGL_3; };
sampler sGL_4		{ Texture = texGL_4; };
sampler sGS_0		{ Texture = texGS_0; };
sampler sGS_1		{ Texture = texGS_1; };
sampler sGS_2		{ Texture = texGS_2; };
sampler sGS_3		{ Texture = texGS_3; };
sampler sGS_4		{ Texture = texGS_4; };

texture2D texYChannel
{ 
    Width = BUFFER_WIDTH; 
    Height = BUFFER_HEIGHT; 
    Format = R8; 
    MipLevels = 6; 
};

sampler sYChannel
{ 
    Texture = texYChannel; 
    MagFilter = LINEAR;
    MinFilter = LINEAR; 
    MipFilter = LINEAR;
};

texture2D texGP
{ 
    Width = BUFFER_WIDTH;   
    Height = BUFFER_HEIGHT;  
    Format = R8; 
};

sampler sGP
{ 
    Texture = texGP; 
    MagFilter = LINEAR; 
    MinFilter = LINEAR; 
    MipFilter = LINEAR;
    AddressU = CLAMP; 
    AddressV = CLAMP; 
    AddressW = CLAMP;
};

/*=============================================================================
=============================================================================*/
float3 from_linear(float3 x)
{
    return lerp(12.92*x, 1.055 * pow(x, 0.4166666666666667) - 0.055, step(0.0031308, x));
}

float3 to_linear(float3 x)
{
    return lerp(x / 12.92, pow((x + 0.055)/(1.055), 2.4), step(0.04045, x));
}

float srgb_luminance(float3 x)
{
    return dot(x, float3(0.2126729, 0.7151522, 0.072175)); 
}

float3 to_rec2020(float3 x)
{
    float3x3 m = float3x3
    (
	    0.6274038936, 0.3292830393, 0.0433130671,
	    0.0690972896, 0.9195403947, 0.0113623157,
	    0.0163914384, 0.0880133079, 0.8955952537
    );

    return mul(m, x);
}

float3 from_rec2020(float3 x)
{
    float3x3 m = float3x3
    (
	    1.6603034854, -0.5875701425, -0.0728900602,
       -0.1243755953,  1.1328344814, -0.0083597372,
       -0.0181122800, -0.1005836085,  1.1187703262
    );

    return mul(m, x);
}

float3 from_hdr(float3 x) 
{ 
    return min(65535.0, 1.0 - exp(-x));
} 

float3 to_hdr(float3 x) 
{
    return max(1e-6, -log(1.0 - x));
}

float uniform_to_triangle(float x)
{
    return (max(-1.0, x * rsqrt(abs(x))) - sign(x)) * 0.5 + 0.5;
}

float get_jitter(float2 p)
{
    float x = frac(p.x * 0.754877666272 + p.y * 0.564189583548);
    return uniform_to_triangle(x * 2.0 - 1.0);
}

/*=============================================================================
=============================================================================*/
void luma_channel(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = srgb_luminance(tex2Dfetch(ReShade::BackBuffer, vpos.xy, 0).rgb);
    if (output < 0.00392157) output += 1e-1; // may be unstable yet..
}

float gaussian(float sigma, float2 offset)
{
    return exp(-dot(offset, offset) / (2.0 * sigma * sigma));
}

int get_effective_radius(float sigma)
{
    float quantile = 2.506628274631;
    return (int)round(quantile * sigma - exp(-sigma * sigma + sqrt(sigma * quantile)) / quantile);
}

float downsample(sampler2D s, float2 coord, float l)
{
    float2 scale = BUFFER_PIXEL_SIZE * (1 << ((int)l + 1));

    float sigma = max(0.707106, sqrt(l * 1.4142135623731));
    
    int k = get_effective_radius(sigma);
    float sum = 0.0;
    float total = 0.0;

    for(int i = -k; i <= k; i++) 
    for(int j = -k; j <= k; j++) 
    {
	    float2 offset = float2(float(i), float(j));
	    float2 sampleUV = coord + offset * scale;

	    float weight = gaussian(sigma, offset);

	    sum += tex2Dlod(s, float4(sampleUV, 0, l)).r * weight;
	    total += weight;
    }

    return sum / total;
}

float upsample(sampler2D s, float2 coord, float l) 
{
    float3 weight;

    float2 scale = BUFFER_PIXEL_SIZE * (1 << ((int)l));
    
    float4 center = tex2Dlod(s, float4(coord, 0, l));
    weight.x = gaussian(0.707106, float2(0, 0));
    
    float4 sides = 
        tex2Dlod(s, float4(coord + float2(-scale.x, 0), 0, l)) +
        tex2Dlod(s, float4(coord + float2( scale.x, 0), 0, l)) +
        tex2Dlod(s, float4(coord + float2(0, -scale.y), 0, l)) +
        tex2Dlod(s, float4(coord + float2(0,  scale.y), 0, l));
    
    weight.y = gaussian(0.707106, float2(1, 0));
    
    float4 corners = 
        tex2Dlod(s, float4(coord + float2(-scale.x, -scale.y), 0, l)) +
        tex2Dlod(s, float4(coord + float2( scale.x, -scale.y), 0, l)) +
        tex2Dlod(s, float4(coord + float2(-scale.x,  scale.y), 0, l)) +
        tex2Dlod(s, float4(coord + float2( scale.x,  scale.y), 0, l));
    
    weight.z = gaussian(0.707106, float2(1, 1));

    float4 sum = center * weight.x + sides * weight.y + corners * weight.z;
    float total = weight.x + 4.0 * weight.y + 4.0 * weight.z;
    
    return sum / total;
}

void dl_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sYChannel, texcoord, 0);
}

void dl_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sGL_0, texcoord, 1);
}

void dl_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sGL_1, texcoord, 2);
}

void dl_3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sGL_2, texcoord,  3);
}

void dl_4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sGL_3, texcoord, 4);
}

void ul_4(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    output = downsample(sGL_4, texcoord, 5);
}

void ul_3(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    float a = upsample(sGL_3, texcoord, 3);
    float b = tex2D(sGS_4, texcoord);
    output = (a + b) / 2.0;
}

void ul_2(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    float a = upsample(sGL_2, texcoord, 2);
    float b = tex2D(sGS_3, texcoord);
    output = (a + b) / 2.0;
}

void ul_1(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    float a = upsample(sGL_1, texcoord, 1);
    float b = tex2D(sGS_2, texcoord);
    output = (a + b) / 2.0;
}

void ul_0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{
    float a = upsample(sGL_0, texcoord, 0);
    float b = tex2D(sGS_1, texcoord);
    output = (a + b) / 2.0;
}

void pyramid(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float output : SV_Target)
{   
    output = tex2D(sGS_0, texcoord) + 
             tex2D(sGS_1, texcoord) + 
             tex2D(sGS_2, texcoord) + 
             tex2D(sGS_3, texcoord) + 
             tex2D(sGS_4, texcoord);
    output /= 5.0;
}

/*=============================================================================
=============================================================================*/
float remap_lowest(float t, int k, int N_max) 
{
    // f∗(t,k) = α^|k|/N_max(t − 1) + 1 (for k < 0)
    float lambda = pow(sqrt(alpha), abs(k) / float(N_max));
    return lambda * (t - 1.0) + 1.0;
}

float remap_higher(float t, int k, int N_max) 
{
    float lambda = pow(sqrt(alpha), k / float(N_max));
    return lambda * t;
}

float current_exposure_range(float t, int k, int N_star, int N, float beta) 
{
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
    return current_exposure_range(remapped, k, N_star, N, beta);
}

float find_best_exposure(float t) 
{
    // k(x) = exp(-(b_u_k(x) - 0.5)²/2σ²), gaussian around ~ 0.05
    return exp(-(t - 0.5) * (t - 0.5) / 0.08); 
}

float contrast_remap(float t, int k, int N_star, int N, float beta) 
{
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
    float lambda_k;
    if (k < 0) {
        lambda_k = pow(sqrt(alpha), abs(k) / float(N_max));
        return lambda_k * contrast_remap(t, k, N_star, N, beta);
    } else {
        lambda_k = pow(sqrt(alpha), k / float(N_max));
        return lambda_k * contrast_remap(t, k, N_star, N, beta);
    }
}

float get_fusion_weights(float t, int k, int N_max, int N_star, int N, float beta) 
{
    float w_e = find_best_exposure(t);
    float w_c = find_best_contrast(t, k, N_max, N_star, N, beta);
    return w_e * w_c;
}

float get_midgrey_value(float x)
{
    const float ev_stops = 8.0;
    float k = log2(midgrey / 0.18);
    return exp2(0.5 * x * k * ev_stops);
}

void exposure_fusion(inout float3 color, float diffuse, int N_max, int N_star, int N, float beta) 
{
    // weighted sum of highlights and shadow fusion result
    float h_sum = 0.0;
    float s_sum = 0.0;

    // total fusion weight
    float h_total = 0.0;
    float s_total = 0.0;

    // basic exposure state for debug
    float h_state = 0.0;
    float s_state = 0.0;

    float exposure_h = solve_exposure(diffuse, -N_star, N_max, N_star, N, beta);
    float exposure_s = solve_exposure(diffuse, 0, N_max, N_star, N, beta);

    [loop]
    for (int k = -N_star; k <= N; k++) 
    {
        float exposure = solve_exposure(diffuse, k, N_max, N_star, N, beta);
        float weight = get_fusion_weights(exposure, k, N_max, N_star, N, beta);

        // process highlights
        [flatten]
        if (k <= 0) {
            h_state += (exposure - exposure_h);
            exposure_h = exposure;
            
            h_sum += exposure * weight;
            h_total += weight;
        }

        // process shadows
        [flatten]
        if (k >= 0) {
            s_state += (exposure - exposure_s);
            exposure_s = exposure;
            
            s_sum += exposure * weight;
            s_total += weight;
        }
    }
    
    float h = h_sum / (h_total + 1e-6);
    float s = s_sum / (s_total + 1e-6);

    float3 a = color * saturate(h * get_midgrey_value(+1));
    float3 b = color / saturate(s * get_midgrey_value(-1));

    if (debug) 
    {
    	float state = (h_state + s_state) / 2;
        state = state > 0.0 ? 0 : state < 0 ? abs(state) : 0;
        float a_1 = state * saturate(h * get_midgrey_value(+1));
        float b_1 = state / saturate(s * get_midgrey_value(-1));
        color = sqrt(a_1 * b_1);
        return;
    }
    
    // geometric mean between fused reference values
    color = min(65535.0, sqrt(a * b));
}

int find_sequence_length(float alpha, float beta, float exposure_bias)
{
    int M = 16; // starts with at least M number of exposures per frame
    bool condition_met = false;
    
    do {
        M++;
        int N_star = (int)round((M - 1) * exposure_bias);
        int N = (M - 1) - N_star;
        int N_max = max(N_star, N);
        
        float t_max_k1 = pow(alpha, 1.0/N_max) * (1.0 + (beta - 1.0)/(M - 1) * (N_star + 1));
        float t_min_k0 = 1.0 + (beta - 1.0)/(M - 1) * N_star - beta;
        
        float t_max_k0 = 1.0 + (beta - 1.0)/(M - 1) * N_star;
        float t_min_km1 = pow(alpha, 1.0/N_max) * (-beta + (beta - 1.0)/(M - 1) * (N_star - 1)) + 1.0;

        condition_met = (t_max_k1 >= t_min_k0) && (t_max_k0 >= t_min_km1);
        
    } while (!condition_met && M < 32);
    
    return M;
}

void main(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float3 output : SV_Target0)
{
    float diffusion = to_linear(tex2D(sGP, texcoord));
    diffusion += diffusion * get_jitter(vpos.xy) * 0.25;

    float3 color = to_linear(tex2Dfetch(ReShade::BackBuffer, vpos, 0).rgb);

    color = to_rec2020(color);
   
    float beta = zone;
    
    int M = find_sequence_length(alpha, beta, re);
    
    int N_star = (int)round(float(M - 1) * re);
    int N = (M - 1) - N_star;
    int N_max = max(N_star, N);
    
    color = to_hdr(color);
    exposure_fusion(color, diffusion, N_max, N_star, N, beta);
    color = from_hdr(color);

    color = from_rec2020(color);
    
    output = from_linear(saturate(color));

    if (debug) 
    {
	    output = 0.0; exposure_fusion(output, diffusion, N_max, N_star, N, beta);
    }
}

/*=============================================================================
=============================================================================*/
technique Rg2Ps_SEF < 
ui_label = "Simulated Exposure Fusion";
ui_tooltip = "			                       Simulated Exposure Fusion \n\n" "A local image processing technique that allows exposure adjustment based on regional image content.\n\n" " - Developed by RG2PS - "; >
{
    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = luma_channel;
	    RenderTarget = texYChannel;
    }

    #define d(i) pass { VertexShader = PostProcessVS; PixelShader = dl_##i; RenderTarget = texGL_##i; }

    d(0)
    d(1)
    d(2)
    d(3)
    d(4)

    #define u(i) pass { VertexShader = PostProcessVS; PixelShader = ul_##i; RenderTarget = texGS_##i; }

    u(0)
    u(1)
    u(2)
    u(3)
    u(4)

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = pyramid;
	    RenderTarget = texGP;
    }

    pass
    {
	    VertexShader = PostProcessVS;
	    PixelShader = main;
    }
}

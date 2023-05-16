#version 450

layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;
layout (set = 2, binding = 0) uniform sampler2D samplerMetallicRoughnessMap;
layout (set = 3, binding = 0) uniform sampler2D samplerNoramlMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;
layout (location = 0) out vec4 outFragColor;
layout (location = 1) out vec4 outColor;

#define PI 3.1415926535897932384626433832795
// Normal Distribution function --------------------------------------
float D_GGX(float dotNH, float roughness)
{
	float alpha = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return (alpha2)/(PI * denom*denom);
}

// Geometric Shadowing function --------------------------------------
float G_SchlicksmithGGX(float dotNL, float dotNV, float roughness)
{
	float r = (roughness + 1.0);
	float k = (r*r) / 8.0;
	float GL = dotNL / (dotNL * (1.0 - k) + k);
	float GV = dotNV / (dotNV * (1.0 - k) + k);
	return GL * GV;
}

// Fresnel function ----------------------------------------------------
vec3 F_Schlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}
vec3 F_SchlickR(float cosTheta, vec3 F0, float roughness)
{
	return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
}
vec3 Uncharted2Tonemap(vec3 x)
{
	float A = 0.15;
	float B = 0.50;
	float C = 0.10;
	float D = 0.20;
	float E = 0.02;
	float F = 0.30;
	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}


void main() 
{

	vec3 albedo = (texture(samplerColorMap, inUV) * vec4(inColor, 1.0)).xyz;
	vec3 F0 = vec3(0.04);
	vec2 metallicRoughness = texture(samplerMetallicRoughnessMap, inUV).xy;
	F0 = mix(F0, albedo, metallicRoughness.x);
	vec3 N = normalize(texture(samplerNoramlMap, inUV).xyz);
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 H = normalize (V + L);
	vec3 color = vec3(0.0);
	float dotNH = clamp(dot(N, H), 0.0, 1.0);
	float dotNV = clamp(dot(N, V), 0.0, 1.0);
	float dotNL = clamp(dot(N, L), 0.0, 1.0);
	float G = G_SchlicksmithGGX(dotNL, dotNV, metallicRoughness.y);
	float D = D_GGX(dotNH, metallicRoughness.y);
	vec3 F = F_SchlickR(dotNV, F0, metallicRoughness.y);
	vec3 R = reflect(L, N);
	vec3 spec = D * F * G / (4.0 * dotNL * dotNV + 0.001);
	vec3 kD = (vec3(1.0) - F) * (1.0 - metallicRoughness.x);
	color += (kD * albedo / PI + spec) * dotNL;
	outColor = vec4(color, 1.0);
	outFragColor = vec4(color, 1.0);
}
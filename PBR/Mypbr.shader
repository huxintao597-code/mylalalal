Shader "Unlit/Myprb"
{
     Properties
    {
        [MainTexture] _BaseMap("漫反射", 2D) = "white" {}
        [MainTexture] _Shelter("遮罩", 2D) = "white" {}
        [MainColor] _BaseColor("基础色", Color) = (1,1,1,1)
        _Roughness("粗糙度", Range(0.0001,1)) = 0.5
        _Metallic("金属度", float) = 0.0
        _SSSMetallic("sss金属度", float) = 0.0
        _subsurface("SSS强度", Range(0.0, 1.0)) = 0.0
        _SSSColor("SSS颜色", Color) = (1, 0.2, 0.1, 1)
        _SSSRadius("SSS半径", Range(0.1, 10.0)) = 1.0
        _SSSPower("SSS幂次", Range(0.1, 5.0)) = 1.0
        _SSSThickness("SSS厚度", Range(0.0, 1.0)) = 0.5
        _SSSTranslucency("透光度", Range(0.0, 2.0)) = 1.0
        _SSSBacklight("背光强度", Range(0.0, 3.0)) = 1.0
        _MetallicGlossMap("金属粗糙度", 2D) = "white" {}
        [HDR] _EmissionColor("Emission Color", Color) = (0,0,0)
        _BumpScale("法线权重", float) = 1.0
        [Normal] _BumpMap("法线", 2D) = "bump" {}
        [Normal] _SSS("SSS遮罩", 2D) = "white" {} // 白色=完全SSS，黑色=无SSS
        _OcclusionMap("AO", 2D) = "white" {}
        _MetallicGlossMap2("粗糙度", 2D) = "white" {}
        [HideInInspector][NoScaleOffset]unity_Lightmaps("unity_Lightmaps", 2DArray) = "" {}
        [HideInInspector][NoScaleOffset]unity_LightmapsInd("unity_LightmapsInd", 2DArray) = "" {}
    }

    SubShader
    {
        Tags {  "Queue" = "Transparent" "RenderType" = "Transparent"  "RenderPipeline" = "UniversalPipeline" }
        Pass
        {
            Name "MyForwardPass"
              Tags {  "Queue" = "Transparent" "RenderType" = "Transparent"  "RenderPipeline" = "UniversalPipeline" }
            ZWrite Off
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            //材质关键词
            #pragma shader_feature _NORMALMAP       //使用法线贴图
            #pragma shader_feature _EMISSION        //开启自发光
            #pragma shader_feature _HIGH_QUALITY_SSS //高质量SSS
            //渲染流水线关键词
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN     //主光开启投射阴影
            #pragma multi_compile _ _SHADOWS_SOFT           //开启软阴影
            //Unity定义的关键词
            #pragma multi_compile _ DIRLIGHTMAP_COMBINED    //开启 lightmap定向模式
            #pragma multi_compile _ LIGHTMAP_ON             //开启 lightmap
            #pragma multi_compile_fog                       //开启雾效

            //有了这个，就不用写那些采样和声明了
            // #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 vertex: POSITION;    //模型空间顶点坐标
                float3 normal: NORMAL;      //模型空间法向量
                float4 tangent: TANGENT;    //模型空间切向量
                float2 uv: TEXCOORD0;       //第一套 uv
                float2 uv2: TEXCOORD1;      //lightmap uv
            };

            struct Varyings
            {
                float4 pos: SV_POSITION;        //齐次裁剪空间顶点坐标
                float2 uv: TEXCOORD0;           //纹理坐标
                float3 normalWS: TEXCOORD1;     //世界空间法线
                float3 viewDirWS: TEXCOORD2;    //世界空间视线方向
                
                #if defined(REQUIRES_WORLD_SPACE_POS_INTERPOLATOR)
                    float3 posWS: TEXCOORD3;    //世界空间顶点位置
                #endif
                
                DECLARE_LIGHTMAP_OR_SH(lightmapUV, vertexSH, 4);   //声明光照贴图的纹理坐标，光照贴图名称、球谐光照名称、纹理坐标索引
                
                #ifdef _NORMALMAP
                    float4 tangentWS: TEXCOORD5;            //xyz是世界空间切向量，w是方向
                #endif
                
                half4 fogFactorAndVertexLight: TEXCOORD6;   //x是雾系数，yzw为顶点光照

                #if defined(REQUIRES_VERTEX_SHADOW_COORD_INTERPOLATOR)
                    float4 shadowCoord: TEXCOORD7;          //阴影坐标
                #endif
            };

            CBUFFER_START(UnityPerMaterial)
            float4 _BaseMap_ST;
            half4 _BaseColor;
            half _Roughness;
            half _Metallic;
            half  _SSSMetallic;
            half _BumpScale;
            float _subsurface;
            half4 _SSSColor;
            half _SSSRadius;
            half _SSSPower;
            half _SSSThickness;
            half _SSSTranslucency;
            half _SSSBacklight;
            CBUFFER_END
            
            TEXTURE2D(_BaseMap);    SAMPLER(sampler_BaseMap);
            TEXTURE2D(_DetailNormalMap);    SAMPLER(sampler_DetailNormalMap);
            TEXTURE2D(_BumpMap);    SAMPLER(sampler_BumpMap);
            TEXTURE2D(_SSS);    SAMPLER(sampler_SSS);
            TEXTURE2D(_MetallicGlossMap);   SAMPLER(sampler_MetallicGlossMap);
             TEXTURE2D(_MetallicGlossMap2);   SAMPLER(sampler_MetallicGlossMap2);
            TEXTURE2D(_OcclusionMap); SAMPLER(sampler_OcclusionMap);
            TEXTURE2D(_Shelter); SAMPLER(sampler_Shelter);
            Varyings vert(Attributes v)
            {
                Varyings o = (Varyings)0;
                VertexPositionInputs vertexInput = GetVertexPositionInputs(v.vertex.xyz);       //获得各个空间下的顶点坐标
                VertexNormalInputs normalInput = GetVertexNormalInputs(v.normal, v.tangent);    //获得各个空间下的法线切线坐标
                float3 viewDirWS = GetCameraPositionWS() - vertexInput.positionWS;              //世界空间视线方向=世界空间相机位置-世界空间顶点位置
                half3 vertexLight = VertexLighting(vertexInput.positionWS, normalInput.normalWS);   //遍历灯光做逐顶点光照（考虑了衰减）
                half fogFactor = ComputeFogFactor(vertexInput.positionCS.z);

                o.uv = TRANSFORM_TEX(v.uv, _BaseMap);   //获得纹理坐标
                o.normalWS = normalInput.normalWS;
                o.viewDirWS = viewDirWS;
                
                #ifdef _NORMALMAP
                    real sign = v.tangent.w * GetOddNegativeScale();
                    o.tangentWS = half4(normalInput.tangentWS.xyz, sign);
                #endif
                
                OUTPUT_LIGHTMAP_UV(v.uv2, unity_LightmapST, o.lightmapUV);  //lightmap uv
                OUTPUT_SH(o.normalWS.xyz, o.vertexSH);  //SH
                
                o.fogFactorAndVertexLight = half4(fogFactor, vertexLight);  //计算雾效

                #if defined(REQUIRES_WORLD_SPACE_POS_INTERPOLATOR)
                    o.posWS = vertexInput.positionWS;
                #endif

                #if defined(REQUIRES_VERTEX_SHADOW_COORD_INTERPOLATOR)
                    o.shadowCoord = GetShadowCoord(vertexInput);
                #endif

                o.pos = vertexInput.positionCS;     //齐次裁剪空间顶点坐标
                return o;
            }
             ///
            /// helper
            /// 
            float3 mon2lin(float3 x)
            {
                return float3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
            }
            float sqr(float x) { return x*x; }

            ///
            /// PBR direct
            ///
            
            float3 compute_F0(float eta)
            {
                return pow((eta-1)/(eta+1), 2);
            }
            float3 F_fresnelSchlick(float VdotH, float3 F0)  // F
            {
                return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
            }
            float3 F_SimpleSchlick(float HdotL, float3 F0)
            {
                return lerp(exp2((-5.55473*HdotL-6.98316)*HdotL), 1, F0);
            }
            
            float SchlickFresnel(float u)
            {
                float m = clamp(1-u, 0, 1);
                float m2 = m*m;
                return m2*m2*m; // pow(m,5)
            }
            float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness)
            {
                return F0 + (max(float3(1.0 - roughness,1.0 - roughness,1.0 - roughness), F0) - F0) * pow(1.0 - cosTheta, 5.0);
            }   

            float GTR1(float NdotH, float a)
            {
                if (a >= 1) return 1/PI;
                float a2 = a*a;
                float t = 1 + (a2-1)*NdotH*NdotH;
                return (a2-1) / (PI*log(a2)*t);
            }
            
            float D_GTR2(float NdotH, float a)    // D
            {
                float a2 = a*a;
                float t = 1 + (a2-1)*NdotH*NdotH;
                return a2 / (PI * t*t);
            }
            
            // X: tangent
            // Y: bitangent
            // ax: roughness along x-axis
            float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
            {
                return 1 / (PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ));
            }
            
            float smithG_GGX(float NdotV, float alphaG)
            {
                float a = alphaG*alphaG;
                float b = NdotV*NdotV;
                return 1 / (NdotV + sqrt(a + b - a*b));
            }

            float GeometrySchlickGGX(float NdotV, float k)
            {
                float nom   = NdotV;
                float denom = NdotV * (1.0 - k) + k;
            
                return nom / denom;
            }
            
          
            
            float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
            {
                return 1 / (NdotV + sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ));
            }

            float3 Diffuse_Burley_Disney( float3 DiffuseColor, float Roughness, float NoV, float NoL, float VoH )
            {
                float FD90 = 0.5 + 2 * VoH * VoH * Roughness;
                float FdV = 1 + (FD90 - 1) * pow(1 - NoV, 5);
                float FdL = 1 + (FD90 - 1) * pow(1 - NoL, 5);
                return DiffuseColor * ((1 / PI) * FdV * FdL);
            }

            // 优化的SSS函数 - 基于物理的次表面散射
            float3 ImprovedSSS(float3 L, float3 V, float3 N, float3 baseColor, float thickness)
            {
                float NdotL = dot(N, L);
                float NdotV = dot(N, V);
                
                // 早期退出 - 背面或视线背面
                if (NdotL < 0.0 || NdotV < 0.0)
                {
                    return 0.0;
                }
                
                // 转换到线性空间
                float3 Cdlin = mon2lin(baseColor);
                
                // 计算半向量
                float3 H = normalize(L + V);
                float LdotH = dot(L, H);
                
                // 基于厚度的透光强度
                float thicknessFactor = lerp(0.5, 1.5, thickness);
                
                // 改进的菲涅尔项
                float FL = SchlickFresnel(NdotL);
                float FV = SchlickFresnel(NdotV);
                
                // 透光核心计算 - 修复发黑问题
                float3 sssResult = 0;
                
                // 1. 正面透光 - 光线从正面进入，从背面散射出来
                float frontTranslucency = pow(1.0 - NdotL, _SSSPower) * _SSSTranslucency;
                float3 frontSSS = baseColor * frontTranslucency * (1.0 - _SSSMetallic);
                
                // 2. 边缘透光 - 在边缘处增强透光效果
                float edgeFactor = 1.0 - abs(dot(normalize(L + N), V));
                edgeFactor = pow(edgeFactor, 2.0);
                float edgeTranslucency = edgeFactor * _SSSTranslucency;
                
                // 3. 背光散射 - 模拟光线从背面穿透的效果
                float backlightFactor = 0;
                if (NdotL > 0.1) // 避免完全正面时的背光
                {
                    // 计算背光角度
                    float3 backLight = normalize(L - 2.0 * dot(L, N) * N);
                    float backlightDot = max(0, dot(backLight, V));
                    backlightFactor = pow(backlightDot, 3.0) * _SSSBacklight;
                }
                
                // 4. 基于粗糙度的散射分布
                float roughnessFactor = 1.0 - _Roughness; // 越光滑越透光
                float scatterDistribution = exp(-_SSSRadius * (1.0 - NdotL));
                scatterDistribution = pow(scatterDistribution, _SSSPower);
                
                // 组合所有SSS项
                sssResult = frontSSS + (edgeTranslucency * baseColor * roughnessFactor) + 
                           (backlightFactor * _SSSColor.rgb * baseColor);
                
                // 应用散射分布
                sssResult *= scatterDistribution;
                
                // 应用SSS颜色调制
                sssResult *= _SSSColor.rgb;
                
                // 确保不会发黑 - 添加最小亮度
                float3 minBrightness = baseColor * 0.1; // 最小保持10%的亮度
                sssResult = max(sssResult, minBrightness);
                
                return sssResult;
            }
            
            // 快速SSS - 性能优化版本，同样修复发黑问题
            float3 FastSSS(float3 L, float3 V, float3 N, float3 baseColor)
            {
                float NdotL = dot(N, L);
                float NdotV = dot(N, V);
                
                if (NdotL < 0.0 || NdotV < 0.0)
                    return 0.0;
                
                float3 Cdlin = mon2lin(baseColor);
                
                // 简化的透光计算
                float viewFactor = 1.0 - NdotV;
                float lightFactor = 1.0 - NdotL;
                
                // 透光强度 - 避免发黑
                float translucency = pow(lightFactor, _SSSPower) * _SSSTranslucency;
                
                // 边缘增强
                float edgeEnhancement = pow(viewFactor, 2.0);
                
                // 背光效果
                float backlight = 0;
                if (NdotL > 0.1)
                {
                    float3 backLight = normalize(L - 2.0 * dot(L, N) * N);
                    float backlightDot = max(0, dot(backLight, V));
                    backlight = pow(backlightDot, 2.0) * _SSSBacklight;
                }
                
                // 组合效果
                float3 sss = (translucency + edgeEnhancement + backlight) * baseColor;
                sss *= _SSSColor.rgb;
                sss *= (1.0 - _SSSMetallic);
                
                // 确保最小亮度
                float3 minBrightness = baseColor * 0.15;
                sss = max(sss, minBrightness);
                
                return sss;
            }

           
            half4 frag(Varyings i) : SV_Target
            {
                //初始化视线
                half3 viewDirWS = SafeNormalize(i.viewDirWS);
                //初始化法线
                half4 n = SAMPLE_TEXTURE2D(_BumpMap, sampler_BumpMap, i.uv);  //采集切线空间法线
                half3 normalTS = UnpackNormalScale(n, _BumpScale);
                #ifdef _NORMALMAP
                    float sgn = i.tangentWS.w;      // should be either +1 or -1
                    float3 bitangent = sgn * cross(i.normalWS.xyz, i.tangentWS.xyz);    //次切线
                    half3x3 tangentToWorld = half3x3(i.tangentWS.xyz, bitangent.xyz, i.normalWS.xyz);   //TBN矩阵
                    i.normalWS = TransformTangentToWorld(normalTS, tangentToWorld);
                #else
                    i.normalWS = i.normalWS;
                #endif
                i.normalWS = NormalizeNormalPerPixel(i.normalWS);
                //初始化金属度粗糙度
                half4 specGloss = SAMPLE_TEXTURE2D(_MetallicGlossMap, sampler_MetallicGlossMap, i.uv); //获取金属粗糙度纹理
                half specGloss2 = SAMPLE_TEXTURE2D(_MetallicGlossMap2,sampler_MetallicGlossMap2,i.uv); //获取糙度纹理
                half metallic = specGloss.r * _Metallic; ///金属度乘强度
                half roughness = specGloss2.r * _Roughness;///粗糙度乘强度
                // half roughness = _Roughness;
                //初始化阴影
                float4 shadowCoord = float4(0, 0, 0, 0);
                #if defined(REQUIRES_VERTEX_SHADOW_COORD_INTERPOLATOR)
                    shadowCoord = i.shadowCoord;
                #elif defined(MAIN_LIGHT_CALCULATE_SHADOWS)
                    shadowCoord = TransformWorldToShadowCoord(i.posWS);
                #endif

                Light mainLight = GetMainLight(shadowCoord);       //获取带阴影的主光
                //金属粗糙度工作流
                float4 albedoAlpha = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, i.uv);      //非导体反照率颜色=baseColor+高光反射颜色
                float3 albedo = albedoAlpha.rgb * _BaseColor.rgb;
                
                
                float3 L = normalize(mainLight.direction);  
                half3 H = normalize(viewDirWS + L);     //半向量

               
                   //return float4(sss, 1);
                float VoH = max(0.001, saturate(dot(viewDirWS, H)));
                float NoV = max(0.001, saturate(dot(i.normalWS, viewDirWS)));
                float NoL = max(0.001, saturate(dot(i.normalWS, L)));
                float NoH = saturate(dot(i.normalWS, H));

                /*   VoH	dot(V, H)	视线与半程向量的对齐程度	菲涅尔反射项计算核心参数
                (Fresnel term)
                NoV	dot(N, V)	表面朝向与视线的夹角余弦	几何遮蔽项分母
                边缘暗化计算
                NoL	dot(N, L)	表面受光强度 (兰伯特余弦定律)	漫反射项权重
                高光项分母
                NoH	dot(N, H)	微表面法线与宏观法线的对齐程度	法线分布函数核心输入
                (如GGX)
                */
                
                half3 radiance = mainLight.color * (mainLight.shadowAttenuation * NoL);     //获取光强
                
                half3 F0 = lerp(half3(0.04, 0.04, 0.04), albedo, metallic);   //F0就是 brdfSpecular，half3(0.04, 0.04, 0.04)是非金属的 F0典型值
                //其实 0.04是由 0.08 * SpecularScale获得的，而 SpecularScale默认值为 0.5
                
                //菲涅尔项 F Schlick Fresnel
                float3 F_Schlick = F0 + (1-F0) * pow(1 - VoH, 5.0);
                float3 Kd = (1-F_Schlick)*(1-metallic);
                float3 brdfDiffuse = albedo * Kd;
                
                //lambert diffuse 
                // float3 diffuseColor = brdfDiffuse * mainLight.color  * NoL;
                float3 diffuseColor = brdfDiffuse * radiance;
                
                // 初始化SSS贴图
                half sssMask = 1.0;
                #ifdef _NORMALMAP
                    sssMask = SAMPLE_TEXTURE2D(_SSS, sampler_SSS, i.uv).r;
                #endif
                
                // 计算改进的SSS
                float3 sss = 0;
                if (_subsurface > 0.0)
                {
                    // 根据性能需求选择SSS算法
                    #if defined(_HIGH_QUALITY_SSS)
                        sss = ImprovedSSS(L, normalize(viewDirWS), normalize(i.normalWS), albedoAlpha.rgb, _SSSThickness);
                    #else
                        sss = FastSSS(L, normalize(viewDirWS), normalize(i.normalWS), albedoAlpha.rgb);
                    #endif
                    
                    // 应用SSS遮罩
                    sss *= sssMask;
                    
                    // SSS作为额外的透光效果叠加，而不是替换漫反射
                    // 这样既保持原有的漫反射，又增加透光效果
                    float3 sssContribution = sss * radiance * _subsurface;
                    diffuseColor += sssContribution;
                }

                //迪士尼 Diffuse
                // float FD90 = 0.5 + 2 * VoH * VoH * roughness;
                // float FdV = 1 + (FD90 -1) * pow(1 - NoV, 5);
                // float FdL = 1 + (FD90 -1) * pow(1 - NoL, 5);
                // float3 diffuseColor =  brdfDiffuse * ((1 / PI) * FdV * FdL);
                // diffuseColor *= mainLight.color * PI * NoL;
                // float3 diffuseColor =  brdfDiffuse  * mainLight.color * NoL;

                
                //法线分布项 D NDF GGX
                float a = roughness * roughness;
                float a2 = a * a;
                float d = (NoH * a2 - NoH) * NoH + 1;
                float D_GGX = a2 / (PI * d * d);


                //几何项 G
                float k = (roughness + 1) * (roughness + 1) / 8;
                float GV = NoV / (NoV * (1-k) + k);
                float GL = NoL / (NoL * (1-k) + k);
                float G_GGX = GV * GL;

                float3 brdf = F_Schlick * D_GGX * G_GGX / (4 * NoV * NoL);
                float3 specularColor = brdf * radiance * PI;

                //间接光 diffuse
                float3 indirectDiffuse = 0;
                //lightmap间接光
                #ifdef LIGHTMAP_ON
                    float3 lm = SampleLightmap(i.lightmapUV, i.normalWS);
                    indirectDiffuse.rgb = lm * albedo * Kd;
                #endif
                
              //  float3 finalDiffuse = lerp(diffuseColor, sss, saturate(_subsurface));
                float3 color =   specularColor + indirectDiffuse+diffuseColor+sss;
                //计算雾效
                //color = MixFog(color, i.fogFactorAndVertexLight.x);
                // return float4(i.normalWS, 1);
                 float4 ALnum = SAMPLE_TEXTURE2D(_Shelter, sampler_Shelter, i.uv);
                return float4(color, 0.11);
            }
            ENDHLSL
        }
        Pass
        {
            Name "MyShadowCaster"
            Tags{"LightMode" = "ShadowCaster"}
            
            ZWrite On
            ZTest LEqual
            Cull[_Cull]
            
            HLSLPROGRAM
            #pragma only_renderers gles gles3 glcore d3d11
            #pragma target 2.0

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local_fragment _ALPHATEST_ON
            #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A

            // -------------------------------------
            // Universal Pipeline keywords

            // This is used during shadow map generation to differentiate between directional and punctual light shadows, as they use different formulas to apply Normal Bias
            #pragma multi_compile_vertex _ _CASTING_PUNCTUAL_LIGHT_SHADOW

            #pragma vertex ShadowPassVertex
            #pragma fragment ShadowPassFragment

            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/ShadowCasterPass.hlsl"
            ENDHLSL
        }
        
    }
}

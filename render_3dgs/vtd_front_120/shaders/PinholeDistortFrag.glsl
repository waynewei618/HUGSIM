#version 400 compatibility

layout( location = 0 ) out vec4 FragColor;

uniform sampler2DRect u_originalScene;
uniform sampler2DRect u_lookupTex;
uniform sampler2DRect u_edgeBlendTex;


vec2 adjustTexCoords(vec2 tc, vec2 texSize) {
    vec2 invTexSize = 1.0/texSize;

    vec2 scaling = vec2(1.0) - 1.0*invTexSize;
    vec2 offset = vec2(0.5)*invTexSize;

    return (tc*scaling + offset);
}

void main( void )
{
    //vec2 sceneTexSize = vec2(textureSize(u_originalScene, 0));
    //vec2 lookupTexSize = vec2(textureSize(u_lookupTex, 0));

    //vec2 tc = adjustTexCoords(gl_TexCoord[0].st/sceneTexSize, lookupTexSize);
    //vec2 texCoord = texture(u_lookupTex, tc*lookupTexSize ).xy;
    vec2 texCoord = texture(u_lookupTex, gl_TexCoord[0].st ).xy;

    vec3 color = texture(u_originalScene, texCoord).rgb;
    //vec4 blendingColor = texture( u_edgeBlendTex,  gl_TexCoord[2].st);

    //FragColor.rgb = color * (1-blendingColor.a);
    FragColor.rgb = color ;

    FragColor.a = 1.0;

}

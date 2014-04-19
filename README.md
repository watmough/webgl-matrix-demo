webgl-matrix-demo
=================

High performance matrix multiplication in Javascript vs WebGL.

Outline

WebGL provides a way to access the high-performance hardware on your graphics card via Javascript in a web browser.

A vertex shader is used to provide vertex information. For this demo, the vertex shader simply passes through the vertexes for the two triangles that are rendered by the Javascript code.

A fragment shader is a bit more interesting. Fragment shaders are called on each pixel being drawn. This allows us to hang a computation off each pixel in the destination image. In this case, it's just a simply matrix multiply term.

To get the source matrices to the fragment shader, we encode them as floating point data in a texture. If float textures are supported, then this is easy, we just make the red channel of the text be the left hand matrix, and the blue channel be the right hand matrix.

For gpus that don't support float textures, we use two RGBA textures and pass the 32-bit float data in the 4 bytes available in each texel.

When the computation is complete, we extract the bytes and we have our matrix product.

Running on a Radeon 6850, the code gets about 23 GFlops, compared with about 0.6 for Javascript on a Core i7 in Firefox 28.

<p><a href="http://watmough.github.io/webgl-matrix-demo/">Click for demo</a>

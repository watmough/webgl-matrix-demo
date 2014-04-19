/*

gpu_matrix.js

summary:
implements multiplication of matrices up to 2048 x 2048 on the gpu by
encoding matrices as 32-bit floats in the Red and Green planes of an
RGB texture.
also supports 2x byte textures for gpus that don't support FLOAT textures.

neat things:
1. a non pre-multiplied RGBA texture is rendered to in IEEE754-format by
the gpu code. this allows us to read floats directly by calling readPixels
then simply switching the buffer from Uint8Array to a Float32Array.
this has been done before, but my code doesn't need ifs or loops. (except
for the zero check)
2. for gpus such as the Samsung Chromebook, which do not have oes_texture_float_extension, 
we can pass float data as a RGBA BYTE texture. In this case we need to either pass two
textures, or double up the size of the passed texture. We also implement code
to munge the bytes back into floats on the gpu.

notes:
accuracy of the floating point seems to be about 8 decimal places, which
seems reasonable.

todo:
single textures of ALPHA could be passed for different sizes.

*/

function gpu_matrix() {};

gpu_matrix.prototype = {

	__gpumatrix__: undefined,

	renderbuffer : undefined,
	framebuffer  : undefined,
	supportsOES_TEXTURE_FLOAT_EXTENSION : true,

	// multiply this into a passed matrix
	multiply: function(matrix) {
		// check conditions
		if (this.c != matrix.c || this.data===undefined || matrix.data===undefined) {
			throw "error: non-matching matrix or missing data";
		}
		
		// run the web gl stuff
		return this._multiply(this,matrix);
	},

	// return string representation of a matrix
	toString: function(radix) {
		if (this.data===undefined)
			return "undefined gpu_matrix";
		var radix = radix || 10;
		var s = "[";
		var idx = 0, n = this.data.length;
		do {
			s += (idx>=this.c && idx %this.c==0) ? "\n" : "";
			s += this.data[idx].toString(radix) + " ";
		} while(++idx<n);
		return s + "]";
	},

	// set row.x.col and array of data matching dimensions
	setData: function(r,c,data) {
		// verify that data if supplied matches r*c
		if(!(data instanceof Array || data.subarray) ||
			(r===undefined || c===undefined) || 
			(data!==undefined && r*c != data.length)) {
			throw "bad data."
			return null;
		}
		this.r = r;
		this.c = c;
		this.data = data;
		return this;
	},
	
	// get data from the matrix as an object
	getData: function() {
		if (this.data!==undefined) {
			return {r:this.r, c:this.c, data:this.data};
		} else {
			throw "missing data.";
		}
		return null;
	},

	// check and initialize webgl context if needed
	_checkinit: function(r,c) {
		// update canvas size
		var canvas = document.getElementById("matrixglcanvas");
		canvas.height = r;
		canvas.width = c;
		if (this.__gpumatrix__===undefined) {
			console.log("getting webgl");
			// get webgl context
			gpu_matrix.prototype.__gpumatrix__ = canvas.getContext("experimental-webgl",
												 { premultipliedAlpha: false, preserveDrawingBuffer: false } );
			if (this.__gpumatrix__===undefined)
				throw "webgl is not supported.";
			// must support float texture
			var ext;
			try {
				ext = gpu_matrix.prototype.__gpumatrix__.getExtension("OES_texture_float");
			} catch(e) {}
			if ( !ext ) {
				console.log("Your webgl does not support OES_texture_float extension.");
			}
		}
		// set viewport to rows, columns
		console.log("this.__gpumatrix__ " + this.__gpumatrix__);
		gpu_matrix.prototype.__gpumatrix__.viewport(0, 0, c, r);
		return this.__gpumatrix__;
	},
	
	_texelsFromMatrices: function(m1,m2,r,c) {
		// dimensions
		var r1 = m1.r, c1=m1.c, r2=m2.r, c2=m2.c;
		var r = Math.max(r1,r2);
		var c = Math.max(c1,c2);
		var texelcount = r*c;
		// get texel data (rgb) as a Float32Array
		texels = new Float32Array(3*texelcount);
		var d1 = m1.data;
		var d2 = m2.data;
		// special case if same dimensions
		if (r1===r2 && c1===c2) {
			// copy m1 to .r and m2 to .g
			var dst = 0, src1=0, src2=0;
			do {
				texels[dst++] = d1[src1++];
				texels[dst++] = d2[src2++];
				dst++;
			} while(--texelcount);
		} else {
			// copy long and short dimensions
			var row=0, col=0;
			var src1 = 0;
			do {
				texels[(row*c1+col)*3] = d1[src1++];
				texels[(col*r2+row)*3+1] = d2[col*r2+row];
				if (col>=c1) {
					col = 0;
					row++;
				}
			} while(--texelcount);
		}
		return texels;
	},
	
	// SUPPORTS FLOAT MATRIX -> RGBA BYTE
	// ### DOES NOT SUPPORT NON-SQUARE MATRICES
	_texelsFromMatrix: function(m,r,c) {
		// dimensions
		var texelcount = m.r*m.c;
		buffer = new ArrayBuffer(4*texelcount);
		// get texel data (rgba) as a Float32Array
		texels = new Float32Array(buffer);
		// copy data to Float32Array, ...
		var dst = 0, src1=0, src2=0;
		do {
			texels[dst++] = m.data[src1++];
		} while(--texelcount);
		// ..., then return as IEEE754 bytes
		return new Uint8Array(buffer);
	},
	
	// bind TEXTURE0 with m1 in .r and m2 in .g (FLOAT RGB texture)
	_bindDualSrcTexture: function(gl,renderer,m1,m2) {
		// get float array data for texture to multiply
		var texels = this._texelsFromMatrices(m1,m2);
		// create the texture from our floats
		var texture = gl.createTexture();
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(	  gl.TEXTURE_2D, texture);
		gl.texImage2D(	  gl.TEXTURE_2D, /*level*/0, gl.RGB, Math.max(m1.c,m2.c), Math.max(m1.r,m2.r), 0,
						  gl.RGB, gl.FLOAT, texels);
		// clamp to edge to support non-power of two textures
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		// don't interpolate when getting data from texture
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		var sampler = gl.getUniformLocation(renderer,"usampler");
		gl.uniform1i(sampler,0);
		return texture;
	},
	
	// bind passed textureUNIT to passed matrix
	_bindSingleSrcTexture: function(gl,renderer,m,textureUNIT,sampler) {
		// get float array data for texture to multiply
		var texels = this._texelsFromMatrix(m);
		// create the texture from our 4 bytes/texel (IEEE754)
		var texture = gl.createTexture();
		gl.activeTexture(textureUNIT);
		gl.bindTexture(	  gl.TEXTURE_2D, texture);
		gl.texImage2D(	  gl.TEXTURE_2D, /*level*/0, gl.RGBA, m.c, m.r, 0,
						  gl.RGBA, gl.UNSIGNED_BYTE, texels);
		// clamp to edge to support non-power of two textures
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		// don't interpolate when getting data from texture
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
		gl.texParameteri( gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		// set up the appropriate sampler
		var sampler = gl.getUniformLocation(renderer,sampler);
		gl.uniform1i(sampler,textureUNIT-gl.TEXTURE0);
		return texture;
	},
	
	// get the canvas to render to
	_createRenderCanvas: function(gl,m1,m2) {
		// Safari readPixels will not work from an 'off-screen' canvas
		// This code probably needs re-factoring, since the canvas should be the 
		// first thing we grab, then the webgl context
		var rendercanvas = document.getElementById("matrixglcanvas");
		return rendercanvas;
	},

	// bind destination texture	
	_createDstTexture: function(gl,rendercanvas) {
		// create and bind texture to render to
		var dstTex = gl.createTexture();
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D,dstTex);
		gl.texImage2D(gl.TEXTURE_2D,/*level*/0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,rendercanvas);
		return dstTex;
	},
	
	// bind a framebuffer, renderbuffer, texture
	_bindFramebuffer: function(gl,dstTex,m1,m2) {
		// create and bind renderbuffer
		this.renderbuffer = this.renderbuffer || gl.createRenderbuffer();
		gl.bindRenderbuffer(gl.RENDERBUFFER,null);
		gl.bindRenderbuffer(gl.RENDERBUFFER,this.renderbuffer);
		gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT16,m2.c,m1.r);
		// create and bind framebuffer
		this.framebuffer = this.framebuffer || gl.createFramebuffer();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
		gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,dstTex,/*level*/0);
		gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,this.renderbuffer);
		return this.framebuffer;
	},

	// build the glslang program to do the matrix multiply
	_buildRenderer: function(gl) {
		// get compiled shaders
		var vertShader = this._getShader( gl, "x-shader/x-vertex", this._vertexShader );
		if (this.supportsOES_TEXTURE_FLOAT_EXTENSION) {
			var fragShader = this._getShader( gl, "x-shader/x-fragment", this._shader_FLOAT );
		} else {
			// use special shader that builds own ieee754
			var fragShader = this._getShader( gl, "x-shader/x-fragment", this._shader_RGBA );
		}
		// link into a program
		var renderer = gl.createProgram();
		gl.attachShader( renderer, vertShader );
		gl.attachShader( renderer, fragShader );
		gl.linkProgram( renderer );
		gl.useProgram( renderer );
		return renderer;
	},

	// setup required to draw a square to our vertex shader and have
	// fragment shader called for each pixel
	_bindVertices: function(gl, renderer) {
		// bind vertices
		var aPos = gl.getAttribLocation(renderer,"aPos");
		var vertexBuffer = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
		var vertices = [-1.0, -1.0,	 0.0, 1.0, -1.0,  0.0, 1.0,	 1.0,  0.0, -1.0,  1.0,	 0.0];
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
		gl.vertexAttribPointer(aPos, /*item size*/3, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(aPos);
		
		// bind texture cords
		var aTex = gl.getAttribLocation(renderer,"aTex");
		var texCoords = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, texCoords);
		var textureCoords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, ];
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(textureCoords), gl.STATIC_DRAW);
		gl.vertexAttribPointer(aTex, /*item size*/2, gl.FLOAT, false, 0, 0);
		gl.enableVertexAttribArray(aTex);
		
		// index to vertices
		var indices = gl.createBuffer();
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indices);
		var vertexIndices = [0, 1, 2, 0, 2, 3];
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(vertexIndices), gl.STATIC_DRAW);
	},

	// set up vars for the shaders
	_bindUniforms: function(gl,renderer,m1,m2) {
		// get var locations
		var length	= gl.getUniformLocation(renderer, "uLength");
		var outR	= gl.getUniformLocation(renderer, "uOutRows");
		var outC	= gl.getUniformLocation(renderer, "uOutCols");
		var stepS	= gl.getUniformLocation(renderer, "uStepS");
		var stepT	= gl.getUniformLocation(renderer, "uStepT");
		// bind length of one multiply run
		gl.uniform1i(length,m1.c);
		// bind output size
		// 3x1 x 1x2  -> 3x2  input and output canvas/texture
		// [2] x [1 1] = [2 2] called for each point in *output* texture
		// [3]			 [3 3]
		// [5]			 [5 5]
		gl.uniform1f(outR,m1.r);
		gl.uniform1f(outC,m2.c);
		// bind step size for input texture
		// 3x10 x 10x2 -> 3x2 output, but requires 10x10 *input* texture
		gl.uniform1f(stepS, 1./Math.max(m1.c,m2.c));
		gl.uniform1f(stepT, 1./Math.max(m1.r,m2.r));
	},

	// multiply m1 x m2
	_multiply: function(m1,m2) {
		// get the basics up and running
		var rawbuffer = new ArrayBuffer(m2.c*m1.r*4);
		var gl = this._checkinit(m1.r,m2.c);
		var renderer = this._buildRenderer(gl);
		if (this.supportsOES_TEXTURE_FLOAT_EXTENSION) {
			this._bindDualSrcTexture(gl,renderer,m1,m2);
		} else {
			this._bindSingleSrcTexture(gl,renderer,m1,gl.TEXTURE0,"usampler1");
			this._bindSingleSrcTexture(gl,renderer,m2,gl.TEXTURE1,"usampler2");
		}
		this._bindUniforms(gl,renderer,m1,m2);
		this._bindVertices(gl,renderer,m1,m2);
		
		// Not sure why I had this in
//		gl.enable(gl.DEPTH_TEST);
//		gl.depthFunc(gl.LEQUAL);
//		gl.clearDepth(1.0);
//		gl.clearColor(0, 0, 0, 0);

//		Following code was enabled, is it needed?
//
//		Apparently not. Probably just some testing code to make sure the calls were valid. 
//		Interestingly, leaving this code in, completely locks up Firefox 28.0, but other
// 		older versions are just fine. NOTE: It locks up on the second press of 'Test'
//		Leaving it in, just in case someone wants to try and understand why FF 28.0 has an issue.

		// Unbind framebuffer and draw

		// draw to default frame buffer
//		gl.activeTexture(gl.TEXTURE0);
//		gl.activeTexture(gl.TEXTURE1);
//		gl.bindFramebuffer(gl.FRAMEBUFFER,null);
//		gl.drawElements(gl.TRIANGLES, /*num items*/6, gl.UNSIGNED_SHORT, 0);

		// Bind framebuffer including texture ad draw

		// draw to result frame buffer
		var rendercanvas = this._createRenderCanvas(gl,m1,m2);
		var dstTex	= this._createDstTexture(gl,rendercanvas);
		var fbuffer = this._bindFramebuffer(gl,dstTex,m1,m2);
		// use the 6 vertex indices to draw triangles
//		gl.bindFramebuffer(gl.FRAMEBUFFER,null);
		gl.bindFramebuffer(gl.FRAMEBUFFER,fbuffer);
		if( gl.checkFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE)
			alert("Error: bound framebuffer is not complete.");
		gl.drawElements(gl.TRIANGLES, /*num items*/6, gl.UNSIGNED_SHORT, 0);

		// extract the product and return in new matrix
		prod = new Uint8Array(rawbuffer);
		gl.readPixels(0,0,m2.c,m1.r,gl.RGBA,gl.UNSIGNED_BYTE,prod);
		return gpu_matrix.create(m1.r,m2.c,new Float32Array(rawbuffer));
	},

	// get shader from script tag
	_getShader: function( gl, shadertype, str ){
		console.log("\n" +str+ "\n")
		// create appropriate type of shader
		var shader;
		if ( shadertype == "x-shader/x-fragment" )
			shader = gl.createShader (gl.FRAGMENT_SHADER);
		else if ( shadertype == "x-shader/x-vertex" )
			shader = gl.createShader(gl.VERTEX_SHADER);
		else {
			throw "unknown shader type " + shaderScript.type;
			return null;
		}
		gl.shaderSource(shader, str);
		gl.compileShader(shader);
		if (gl.getShaderParameter(shader, gl.COMPILE_STATUS) == 0)
			alert(gl.getShaderInfoLog(shader));
		return shader;
	},

	_vertexShader : 
"   // vertex shader for a single quad \n"+
"    // work is performed based on the texels being passed \n"+
"    // through to the texture shader. \n"+
"#ifdef GL_ES \n"+
"	precision highp float; \n"+
"#endif \n"+
"	attribute vec3 aPos; \n"+
"	attribute vec2 aTex; \n"+
"	varying vec2   vTex; \n"+
"	void main(void) \n"+
"	{ \n"+
"		// just pass the position and texture coords \n"+
"		gl_Position = vec4(aPos, 1.0); \n"+
"		vTex = aTex; \n"+
"	}",

	_shader_FLOAT :
"    // fragment shader that calculates the sum of the passed row and " +
"    // column (texture coord). \n" +
"    // we loop over the row and column and sum the product. \n" +
"    // product is then rendered to 32-bit IEEE754 floating point in the \n" +
"    // output RGBA canvas. \n" +
"    // readPixel is used to read the bytes. \n" +
"#ifdef GL_ES \n" +
"	precision highp float; \n" + 
"#endif \n" + 
"\n" + 
"	varying vec2	  vTex;         // row, column to calculate \n" + 
"	uniform sampler2D usampler;		// left in .r, right in .g \n" + 
"	uniform int		  uLength;      // r1xc1.r2xc2 => product has r2 (or c1) terms \n" + 
"	uniform float	  uStepS;       // increment across source texture \n" + 
"	uniform float	  uStepT;       // increment down source texture \n" + 
"	uniform float	  uOutRows;     // size of output in rows \n" + 
"	uniform float	  uOutCols;     // size of output in columns \n" + 
"	 \n" + 
"	// sum row r x col c \n" + 
"	float sumrowcol(float row, float col) { \n" + 
"		float sum = 0.;             // sum \n" + 
"		float ss = 0.;              // column on source texture \n" + 
"		float tt = 0.;              // row on source texture \n" + 
"		float r = row*uStepT;       // moving texture coordinate \n" + 
"		float c = col*uStepS;       // moving texture coordinate \n" + 
"		for (int pos=0 ; pos<2048 ; ++pos) { \n" + 
"			if(pos>=uLength) break; // stop when we multiple a row by a column \n" + 
"			float m1 = texture2D(usampler,vec2(ss,r)).r; \n" + 
"			float m2 = texture2D(usampler,vec2(c,tt)).g; \n" + 
"			sum += (m1*m2); \n" + 
"			ss += uStepS; \n" + 
"			tt += uStepT; \n" + 
"		} \n" + 
"		return sum; \n" + 
"	} \n" + 
"	 \n" + 
"	void main(void) { \n" + 
"		 \n" + 
"		// get the implied row and column from .s and .t of passed texel \n" + 
"		float col = floor((vTex.s*uOutRows)); \n" + 
"		float row = floor((vTex.t*uOutCols));    \n" + 
"\n" + 
"		// sum row x col for the passed pixel \n" + 
"		float v = sumrowcol(row,col); \n" + 
"\n" + 
"		// Render to IEEE 754 Floating Point \n" + 
"		if (v==0.) { \n" + 
"			gl_FragColor = vec4(0.,0.,0.,0.); \n" + 
"			return; \n" + 
"		} \n" + 
"		float a = abs(v);                           // encode absolute value + sign \n" + 
"		float exp = floor(log2(a));                 // number of powers of 2 \n" + 
"		float mant = (a * pow(2.,23.-exp));         // multiply to fill 24 bits (implied leading 1) \n" + 
"		float mant1 = floor(mant / 256. / 256.);    // first 8 bits of mantissa \n" + 
"		float mant2 = mod(floor(mant / 256.),256.); // second 8 bits \n" + 
"		float mant3 = mod(mant,256.);               // third 8 bits \n" + 
"		 \n" + 
"		highp float sign = 128.-128.*(a/v);			// sign bit is 256 or 0 \n" + 
"		highp float e = (sign+exp+127.)/510.;		// exponent and sign \n" + 
"		highp float m1 = (mant1-(128.*(1.-mod(exp+127.,2.))))/255.; // handle leading bit \n" + 
"		highp float m2 = (mant2)/255.;				// middle part \n" + 
"		highp float m3 = (mant3+.5)/255.;			// scale to 0 - 255 \n" + 
"		gl_FragColor = vec4(m3,m2,m1,e);			// output an IEEE754 32-bit floating point number \n" + 
"	} ",
 
	_shader_RGBA :
"	// EXPERIMENTAL: READ FLOAT DATA FROM RGBA BYTES IN IEEE754 \n" + 
"    // fragment shader that calculates the sum of the passed row and \n" + 
"    // column (texture coord). \n" + 
"    // we loop over the row and column and sum the product. \n" + 
"    // product is then rendered to 32-bit IEEE754 floating point in the \n" + 
"    // output RGBA canvas. \n" + 
"    // readPixel is used to read the bytes. \n" + 
"#ifdef GL_ES \n" + 
"	precision highp float; \n" + 
"#endif \n" + 
"\n" + 
"	varying vec2	  vTex;         // row, column to calculate \n" + 
"	uniform sampler2D usampler1;	// LEFT \n" + 
"	uniform sampler2D usampler2;	// RIGHT \n" + 
"	uniform int		  uLength;      // r1xc1.r2xc2 => product has r2 (or c1) terms \n" + 
"	uniform float	  uStepS;       // increment across source texture \n" + 
"	uniform float	  uStepT;       // increment down source texture \n" + 
"	uniform float	  uOutRows;     // size of output in rows \n" + 
"	uniform float	  uOutCols;     // size of output in columns \n" + 
"\n" + 
"	/* \n" + 
"	// javascript decrypt ieee754 2013/02/08 \n" + 
"	mant3 = prod[src+0]; \n" + 
"	mant2 = prod[src+1]; \n" + 
"	bit = Math.floor(prod[src+2]/128.)*128; \n" + 
"	mant1 = prod[src+2]-(bit-128.); \n" + 
"	exp = ((prod[src+3] % 128)*2) + bit/128.; \n" + 
"	sgn = Math.floor(prod[src+3]/128.); \n" + 
"	f = mant1*256*256 + mant2*256 + mant3; \n" + 
"	f = f * Math.pow(2,(exp-150))*(1-2*sgn); \n" + 
"	*/ \n" + 
"\n" + 
"	float toIEEE754(vec4 bytes) { \n" + 
"		// RETURN AN IEEE754 FLOAT FROM 4 BYTES \n" + 
"		// GET BYTES \n" + 
"		float byte0 = bytes.r*255.; \n" + 
"		float byte1 = bytes.g*255.; \n" + 
"		float byte2 = bytes.b*255.; \n" + 
"		float byte3 = bytes.a*255.; \n" + 
"		// COMPUTE \n" + 
"		float mant3 = byte0; \n" + 
"		float mant2 = byte1; \n" + 
"		float bitv = floor(byte2/128.)*128.; \n" + 
"		float mant1 = byte2-(bitv-128.); \n" + 
"		float expv = (mod(byte3,128.))*2. + bitv/128.; \n" + 
"		float sgnv = floor(byte3/128.); \n" + 
"		float f = (mant1*256.*256.) + (mant2*256.) + mant3; \n" + 
"		f = f * pow(2.,(expv-150.))*(1.-2.*sgnv); \n" + 
"		return f; \n" + 
"	} \n" + 
"\n" + 
"	// sum row r x col c \n" + 
"	float sumrowcol(float row, float col) { \n" + 
"		float sum = 0.;             // sum \n" + 
"		float ss = 0.;              // column on source texture \n" + 
"		float tt = 0.;              // row on source texture \n" + 
"		float r = row*uStepT;       // moving texture coordinate \n" + 
"		float c = col*uStepS;       // moving texture coordinate \n" + 
"		for (int pos=0 ; pos<2048 ; ++pos) { \n" + 
"			if(pos>=uLength) break; // stop when we multiple a row by a column \n" + 
"			float m1 = toIEEE754(texture2D(usampler1,vec2(ss,r))); \n" + 
"			float m2 = toIEEE754(texture2D(usampler2,vec2(c,tt))); \n" + 
"// used for verifying correct sampling of texture" +
"//			return m1; \n" + 
"//			return float(texture2D(usampler1,vec2(ss,r)).r*255.); \n" + 
"			sum += (m1*m2); \n" + 
"			ss += uStepS; \n" + 
"			tt += uStepT; \n" + 
"		} \n" + 
"		return sum; \n" + 
"	} \n" + 
"	 \n" + 
"	void main(void) { \n" + 
"		 \n" + 
"		// get the implied row and column from .s and .t of passed texel \n" + 
"		float col = floor((vTex.s*uOutRows)); \n" + 
"		float row = floor((vTex.t*uOutCols));    \n" + 
"\n" + 
"		// sum row x col for the passed pixel \n" + 
"		float v = sumrowcol(row,col); \n" + 
"\n" + 
"		// Render to IEEE 754 Floating Point \n" + 
"		if (v==0.) { \n" + 
"			gl_FragColor = vec4(0.,0.,0.,0.); \n" + 
"			return; \n" + 
"		} \n" + 
"		float a = abs(v);                           // encode absolute value + sign \n" + 
"		float exp = floor(log2(a));                 // number of powers of 2 \n" + 
"		float mant = (a * pow(2.,23.-exp));         // multiply to fill 24 bits (implied leading 1) \n" + 
"		float mant1 = floor(mant / 256. / 256.);    // first 8 bits of mantissa \n" + 
"		float mant2 = mod(floor(mant / 256.),256.); // second 8 bits \n" + 
"		float mant3 = mod(mant,256.);               // third 8 bits \n" + 
"		 \n" + 
"		highp float sign = 128.-128.*(a/v);			// sign bit is 256 or 0 \n" + 
"		highp float e = (sign+exp+127.)/510.;		// exponent and sign \n" + 
"		highp float m1 = (mant1-(128.*(1.-mod(exp+127.,2.))))/255.; // handle leading bit \n" + 
"		highp float m2 = (mant2)/255.;				// middle part \n" + 
"		highp float m3 = (mant3+.5)/255.;			// scale to 0 - 255 \n" + 
"		gl_FragColor = vec4(m3,m2,m1,e);			// output an IEEE754 32-bit floating point number \n" + 
"	} "

};

gpu_matrix.create = function(r,c,data) {
	var M = new gpu_matrix();
	return M.setData(r,c,data);
};





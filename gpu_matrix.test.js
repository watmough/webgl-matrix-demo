/*

gpu_matrix.test.js

basic testing for multiplying matrices together on the gpu.

*/

function gpu_matrix_test() {
    
    // work up through different sizes of matrices
    var log = "";
    for (var sz=16; sz<=2048 ; sz*=2) {
        // set up

        // Can use NPOT sizes ok
        var n = sz, matrixSide = sz;

        // create a few test matrices
        mmCount     = matrixCount(matrixSide);
        mmRandom    = matrixRandom(matrixSide);
        mmIdentity  = matrixIdentity(matrixSide);

        // pick a couple to test with
        var mm1 = mmRandom;
        var mm2 = mmRandom;

        // calculate number of ops
        var ops = 2.0*n*n*n - n*n;

        // multiply them on gpu
        var start = (new Date()).getTime();
        result = mm1.multiply(mm2);
        var dur = (new Date()).getTime()-start;
        var gflops = Math.round(ops / (dur*10000))/100;
        log += "\nMultiplied " + mm1.r + "x" + mm1.c + " x " + mm2.r + "x" + mm2.c + " matrices in " + dur +
              " msec. (GPU) " + gflops + " GFlops \n";
        
        // multiply smaller ones in js and check result
        if (n<=512) {
            var start = (new Date()).getTime();
            resultjs = slowMult(mm1,mm2);
            var dur = (new Date()).getTime()-start+1;
            var gflops = Math.round(ops / (dur*10000))/100;
            log += "Multiplied "+mm1.r+"x"+mm1.c+" x "+mm2.r+"x"+mm2.c+" matrices in "+dur+
                   " msec. (Javascript) " + gflops + " GFlops \n";

            // check the result
            var err = 0, errterm = 0, errmax = 0, errjs = 0, errgpu = 0;
            var d1 = result.data;
            var d2 = resultjs.data;
            var sumjs = matrixSummate(resultjs);
            for (var ii=0;ii<d1.length;++ii) {             // for all data
                errterm = Math.abs(d1[ii]-d2[ii]);
                if (errterm>=errmax) {
                    errmax = errterm;
                    termjs = d2[ii];
                    termgpu = d1[ii];
                }
                err += errterm;//Math.abs(errterm);
            }
            log +=  "Absolute error:    " + err + 
                    "\nOver total sum:    " + sumjs +
                    "\nFraction of sum:   " + err/sumjs +
                    "\nMax err in a term: " + errmax + " (" + termgpu + " vs " + termjs + " js)" +
                    "\nMultiply ops:      " + ops + "\n";
        }
        else
            log += "Skipped for Javascript\n" +
                    "Multiply ops:      " + ops + "\n";
    }
    // log the times
    console.log(log);
    return log;
}

// Misc Functions

// identity matrix
function matrixIdentity(n) {
    var m = [];
    for(var i=0;i<n;++i) {
        for(var j=0;j<n;++j) {
            if(i==j) m.push(1); else m.push(0); 
        }
    }
    return gpu_matrix.create(n,n,m);
}

// random matrix
function matrixRandom(n) {
    m = [];
    for(var i=0;i<n*n;++i) {
        m.push(Math.random()*1);//000000000000);
    }
    return gpu_matrix.create(n,n,new Float32Array(m));
}

// elements numbered ascending
function matrixCount(n) {
    m = [];
    for(var i=0;i<n*n;++i) {
        m.push(i);
    }
    return gpu_matrix.create(n,n,new Float32Array(m));
}

function matrixSummate(m) {
    var sum = 0;
    var size = m.r*m.c;
    for(var i=0;i<size;++i) {
        sum += m.data[i];
    }
    return sum;
}

// javascript matrix multiply
function slowMult(m1,m2) {
    var r1 = m1.r|0;
    var c1 = m1.c|0;
    var r2 = m2.r|0;
    var c2 = m2.c|0;
    // new matrix data r1xc2
    var d1 = m1.data;
    var d2 = m2.data;
    var data = new Float32Array(r1*c2);
    for (var ii=0|0;ii<r1;++ii) {             // for each row in product
        for (var jj=0|0;jj<c2;++jj) {         // for each column in product
            var sum=+0;
            for (var kk=0|0;kk<c1;++kk) {     // for each item in sum - across m1, down m2
                sum += d1[ii*c1+kk]*d2[jj+kk*c2];
            }
            data[ii*c2+jj] = sum;
        }
    }
    return gpu_matrix.create(r1,c2,data);
}




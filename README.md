# bilateral_filter

Image smoothing and denoising is a very important part of almost all computer vision and graphics algorithms. The issue with using basic Gaussian blurs or other linear filters are that they may cause us to lose important structural data in our image (think about we would lose our strong edges if we smoothed a checkerboard pattern with a basic Gaussian blur). In order to maintain edges, the bilateral filter assigns weights to it's neighbors based not only on spatial distance but also on intensity distances as well. 


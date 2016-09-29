
#include <chrono>
#include "util.h"
#include <png++/png.hpp>

#define CLAMP_FLOAT(p) ( MIN(1.0,MAX(0.0,p)) )
#define CLAMP_INT(p) ( MIN(255,MAX(0,p)) )
#define CLAMP(val,min,max) ( MIN(max,MAX(min,val)) )
#define PIXEL(r,g,b) ( png::rgba_pixel(CLAMP_INT(r),CLAMP_INT(g),CLAMP_INT(b),255) )

using std::cout;
using std::endl;
using std::string;

double pixel_to_grayscale(png::rgba_pixel p) {
    return ((double)p.red)*0.299+((double)p.green)*0.587+((double)p.blue)*0.114;
}

void usage() {
    fprintf(stderr, "\n"
                    "usage: main <input_image>.png <output_image>.png <options> \n"
                    "\n"
                    "This is an implementation of the classic bilateral filter algorithm. The bilateral filter is an edge-preserving smoothing algoirthm that uses intensity weights in addition to spatial weights in order to preserve edges. \n"
                    "\n"
                    "\n"
                    "Required Parameters: \n"
                    "\n"
                    "    <input_image>.png: This is the file name of the input image. Must be a .png file. \n"
                    "\n"
                    "    <output_image>.png: This is the file name of the output image. Must be a .png file. \n"
                    "\n"
                    "\n"
                    "Options: \n"
                    "\n" 
                    "    -gaussian_kernel_sigma <gaussian_kernel_sigma>: Gaussian weights based on spatial distance are used to determine the impact of nearby pixels. This parameter determines the weights of nearby pixels. \n"
                    "\n"
                    "    -intensity_sigma <intensity_sigma>: Gaussian weights based on intensity difference are used to determine the impact of nearby pixels. This parameter determines the weights of nearby pixels. \n"
                    "\n"
                    "\n"
           );
    exit(1);
}

int main(int argc, char* argv[]) {
    
    if (argc < 3) { 
        usage(); 
    }
    
    /* Parameters */
    static const double gaussian_kernel_sigma = atof(get_command_line_param_val_default_val(argc, argv, "-gaussian_kernel_sigma", "2"));
    static const int gaussian_kernel_dim = 1+2*(int)(gaussian_kernel_sigma*1.5); // To get about a 99% confidence interval
    static const double intensity_sigma = atof(get_command_line_param_val_default_val(argc, argv, "-intensity_sigma", "5"));
    char* input_image_name = argv[1];
    char* output_image_name = argv[2];
    
    png::image< png::rgba_pixel > I(input_image_name); // input image
    static const unsigned I_height = I.get_height();
    static const unsigned I_width = I.get_width();
    png::image< png::rgba_pixel > J(I_width, I_height); // output image
    
    cout << "Running bilateral filter on " << input_image_name << " (" << I_width*I_height << " pixels)." << endl << endl;
    
    Array<double> kernel;
    get_gaussian_kernel(gaussian_kernel_dim, gaussian_kernel_sigma, kernel);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int y = 0; y < I_height; ++y) {
        for (int x = 0; x < I_width; ++x) {
            double final_red = 0;
            double final_green = 0;
            double final_blue = 0;
            double normalization_term = 0;
            double center_px_lum = pixel_to_grayscale(I[y][x]);
            
            for (int r=-kernel.height()/2; r<kernel.height()/2+1; r++) {
                for (int c=-kernel.width()/2; c<kernel.width()/2+1; c++) {
                    
                    auto current_pixel = I[CLAMP(y+r,0,I.get_height()-1)][CLAMP(x+c,0,I.get_width()-1)];
                    double current_px_lum = pixel_to_grayscale(current_pixel);
                    double spatial_weight = kernel(r+kernel.height()/2, c+kernel.height()/2);
                    double intensity_weight = G(fabs(current_px_lum-center_px_lum), intensity_sigma);
                    
                    final_red += current_pixel.red*spatial_weight*intensity_weight;
                    final_green += current_pixel.green*spatial_weight*intensity_weight;
                    final_blue += current_pixel.blue*spatial_weight*intensity_weight;
                    
                    normalization_term += spatial_weight*intensity_weight;
                }   
            }
            J[y][x] = PIXEL((unsigned int)(final_red/normalization_term), (unsigned int)(final_green/normalization_term), (unsigned int)(final_blue/normalization_term));
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    J.write( output_image_name );
    
    cout << "Total Run Time: " << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_time-start_time).count()) / (pow(10.0,9.0)) << " seconds." << endl;
    
    return 0;
}


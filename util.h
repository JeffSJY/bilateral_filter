
#include <vector>
#include <cmath>
#include <string.h>
#include <iostream>
#include <ostream>
#include <cassert>
#include <cstdio>

#define SQUARE(x) ((x)*(x))
#define MOD(a,b) ( ((((int)a)%((int)b))+((int)b))%((int)b) )
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

#define PI (3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709)

#define DEBUG BUILD_DEBUG
#define ASSERT(x, msg) if (DEBUG && !(x)) { fprintf(stderr, "Assertion failed in %s(%d): %s\n", __FILE__, __LINE__, msg); assert(false); exit(1); }

#define TEST(x) cout << (#x) << ": " << x << endl;
#define PRINT(x) cout << x << endl;
#define NEWLINE cout << endl; fflush(stdout);

#define CHAR_STAR_EQUAL(a,b) !strcmp(a,b)

using std::string;
using std::to_string;
using std::ostream;
using std::vector;
using std::cout;
using std::endl;

double G(const double &x, const double &sigma){
    return exp(-(x*x)/(2*SQUARE(sigma))) / sqrt(2*PI*SQUARE(sigma));
}

double G(const double &x, const double &y, const double &sigma){
    return exp(-(SQUARE(y)+SQUARE(x))/(2*SQUARE(sigma))) / (2*PI*SQUARE(sigma));
}

double distance_from_origin(const double &x, const double &y){
    return sqrt( SQUARE(x) + SQUARE(y) );
}

template<class real>
class Array {
    public:
        real *data;
        vector<int> sizes;
        vector<int> stride;
        int nelems;
        
        void resize(const vector<int> &sizes_) {
            if (sizes == sizes_) { return; }
            
            delete[] data;
            sizes = sizes_;
            
            stride.resize(sizes.size());
            nelems = 1;
            for (int i = stride.size()-1; i >= 0; i--) {
                stride[i] = nelems;
                nelems *= sizes[i];
            }
            data = new real[nelems];
        }
        
        void assign(const Array &other) {
            resize(other.sizes);
            #pragma omp parallel for
            for(unsigned i=0; i < nelems; i++){
                data[i] = other.data[i];
            }
        }
        
        Array() :data(NULL) {
            resize(vector<int>{1});
        }
        
        Array(const Array &other) :data(NULL) {
            assign(other);
        }
        
        Array(const vector<int> &sizes_) :data(NULL) {
            resize(sizes_);
        }
        
        ~Array() {
            delete[] data;
        }
        
        void clear(const real &val=0) {
            #pragma omp parallel for
            for(unsigned i=0; i<nelems; i++){
                data[i]=val;
            }
        }
        
        real sum() {
            real sum = 0;
            for(unsigned i=0; i<nelems; i++){
                sum += data[i];
            }
            return sum;
        }
        
        real product() {
            real product = 0;
            for(unsigned i=0; i<nelems; i++){
                product *= data[i];
            }
            return product;
        }
        
        Array& normalize() {
            real sum = this->sum();
            #pragma omp parallel for
            for(unsigned i=0; i<nelems; i++){
                data[i] /= sum;
            }
            return (*this);
        }
        
        int height() const {
            ASSERT(sizes.size() >= 2, "Need at least 2 dimensions to get height");
            return sizes[0];
        }

        int width() const {
            ASSERT(sizes.size() >= 2, "Need at least 2 dimensions to get width");
            return sizes[1];
        }

        int channels() const {
            return (sizes.size()<3) ? 1 : sizes[2];
        }
        
        int dimensions() const {
            return sizes.size();
        }
        
        string str() const {
            string ans("[");
            if (dimensions() == 1) {
                for (int i = 0; i < nelems; i++) {
                    ans += to_string(data[i]);
                    if (i < nelems - 1) { ans += ", "; }
                }
            } else if (dimensions() == 2) {
                for (int y = 0; y < height(); y++) {
                    ans += "[";
                    for (int x = 0; x < width(); x++) {
                        ans += to_string((*this)(y,x));
                        if (x < width()-1) { ans += ", "; }
                    }
                    ans += "]";
                    if (y < height()-1) { ans += ",\n "; }
                }
            }else if (dimensions() == 3) {
                for (int z = 0; z < channels(); z++) {
                    ans += "\nChannel "+to_string(z)+":\n ";
                    for (int y = 0; y < height(); y++) {
                        ans += "[";
                        for (int x = 0; x < width(); x++) {
                            ans += to_string((*this)(y,x,z));
                            if (x < width()-1) { ans += ", "; }
                        }
                        ans += "]";
                        if (y < height()-1) { ans += ",\n "; }
                    }
                    ans += "\n";
                }
            } else {
                ASSERT(false, "Array::str() not implement for dimension > 3");
            }
            ans += "]";
            return ans;
        }
        
        real& operator()(int v0) const {
            ASSERT(sizes.size() == 1, (string("1D lookup in array of dimensionality")+to_string(sizes.size())).c_str()); 
            ASSERT(v0 >= 0 && v0 < sizes[0], (string("1D lookup out of bounds (index=")+to_string(v0)+", length="+to_string(sizes[0])+")").c_str()); 
            return data[v0];
        }
        
        real& operator()(int v0, int v1) const {
            ASSERT(sizes.size() == 2, (string("2D lookup in array of dimensionality")+to_string(sizes.size())).c_str()); 
            ASSERT(v0 >= 0 && v0 < sizes[0] && v1 >= 0 && v1 < sizes[1], (string("2D lookup out of bounds (row=")+to_string(v0)+", column="+to_string(v1)+", height="+to_string(sizes[0])+", width="+to_string(sizes[1])+")").c_str()); 
            return data[v0*stride[0]+v1];
        }
        
        real& operator()(int v0, int v1, int v2) const {
            ASSERT(sizes.size() == 3, (string("3D lookup in array of dimensionality")+to_string(sizes.size())).c_str());
            ASSERT(v0 >= 0 && v0 < sizes[0] && v1 >= 0 && v1 < sizes[1] && v2 >= 0 && v2 < sizes[2], (string("2D lookup out of bounds (row=")+to_string(v0)+", column="+to_string(v1)+", channel="+to_string(v1)+", height="+to_string(sizes[0])+", width="+to_string(sizes[1])+", num_channels="+to_string(sizes[2])+")").c_str()); 
            return data[v0*stride[0]+v1*stride[1]+v2];
        }
};

template<class real>
ostream& operator<<(ostream& os, const Array<real>& obj) {
    os << obj.str();
    return os;
}

void get_gaussian_kernel(const int &dim, const double &sigma, Array<double> &kernel) {
    ASSERT(MOD(dim,2)==1, "Gaussian kernel must be of odd dimension");
    kernel.resize(vector<int>{dim,dim});
    
    #pragma omp parallel for
    for(int y=0; y<dim; y++) {
        for(int x=0; x<y+1; x++) {
            kernel(y,x) = G(y-dim/2, x-dim/2, sigma);
            if (x != y) {
                kernel(x,y) = kernel(y,x);
            }
        }
    }
    kernel.normalize();
}

const char* get_command_line_param_val_default_val(const int &argc, char** &argv, const char* param_option, const char* default_val) {
    for (int i=0;i<argc; i++) {
        if (CHAR_STAR_EQUAL(argv[i],param_option)) {
            if (i+1 != argc) {
                return argv[i+1];
            } else {
                return default_val;
            }
        }
    }
    return default_val;
}


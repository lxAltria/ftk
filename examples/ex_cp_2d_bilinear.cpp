#include <mutex>
#include <ftk/numeric/print.hh>
#include <ftk/numeric/cross_product.hh>
#include <ftk/numeric/vector_norm.hh>
#include <ftk/numeric/linear_interpolation.hh>
#include <ftk/numeric/bilinear_interpolation.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/inverse_bilinear_interpolation_solver.hh>
#include <ftk/numeric/gradient.hh>
#include <ftk/algorithms/cca.hh>
#include <ftk/geometry/cc2curves.hh>
#include <ftk/geometry/curve2tube.hh>
#include <ftk/ndarray.hh>
#include <unordered_map>
#include <vector>
#include <queue>
#include <fstream>

// const int DW = 128, DH = 128;// the dimensionality of the data is DW*DH
int DW = 128, DH = 128;// the dimensionality of the data is DW*DH

ftk::ndarray<float> grad;

struct critical_point_t {
  double mu[3]; // interpolation coefficients
  double x[2]; // the coordinates of the critical points
  double v;
  int type;
  size_t simplex_id;
  critical_point_t(){}
  critical_point_t(const double * mu_, const double * x_, const double v_, const int t_){
    for(int i=0; i<3; i++) mu[i] = mu_[i];
    for(int i=0; i<2; i++) x[i] = x_[i];
    v = v_;
    type = t_;
  }
};
 
// std::map<hypermesh::regular_simplex_mesh_element, critical_point_t> critical_points;
std::unordered_map<int, critical_point_t> critical_points;

#define MAX_SIZE 10000
struct pair_hash{
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2>& p) const{
    return p.first * MAX_SIZE + p.second;
  }
};
std::unordered_map<std::pair<int, int>, double, pair_hash> ebs;

#define DEFAULT_EB 1
#define SINGULAR 0
#define ATTRACTING 1 // 2 real negative eigenvalues
#define REPELLING 2 // 2 real positive eigenvalues
#define SADDLE 3// 1 real negative and 1 real positive
#define ATTRACTING_FOCUS 4 // complex with negative real
#define REPELLING_FOCUS 5 // complex with positive real
#define CENTER 6 // complex with 0 real

std::string get_critical_point_type_string(int type){
  switch(type){
    case 0:
      return "SINGULAR";
    case 1:
      return "ATTRACTING";
    case 2:
      return "REPELLING";
    case 3:
      return "SADDLE";
    case 4:
      return "ATTRACTING_FOCUS";
    case 5:
      return "REPELLING_FOCUS";
    case 6:
      return "CENTER";
    default:
      return "INVALID";
  }
}

int bilinear_critical_point_type(double u[4], double v[4], double x[2]){
  double  f00 = u[0], f10 = u[1], f01 = u[3], f11 = u[2], 
          g00 = v[0], g10 = v[1], g01 = v[3], g11 = v[2];
  double  A0 = f00 - f10 - f01 + f11,  // Axy + Bx + Cy + D = 0
          B0 = f10 - f00, 
          C0 = f01 - f00, 
          D0 = f00,
          A1 = g00 - g10 - g01 + g11, 
          B1 = g10 - g00, 
          C1 = g01 - g00, 
          D1 = g00; 
  double J[2][2] = {{A0 * x[1] + B0, A0 * x[0] + C0}, {A1 * x[1] + B1, A1*x[0] + C1}}; // jacobian  
  // if(fabs(A0*x[0]*x[1] + B0*x[0] + C0*x[1] + D0) > 1e-6){
  //   std::cout << A0*x[0]*x[1] + B0*x[0] + C0*x[1] + D0 << std::endl;
  //   exit(0);
  // }
  // if(fabs(A1*x[0]*x[1] + B1*x[0] + C1*x[1] + D1) > 1e-6){
  //   std::cout << A1*x[0]*x[1] + B1*x[0] + C1*x[1] + D1 << std::endl;
  //   exit(0);
  // }
  int cp_type = 0;
  std::complex<double> eig[2];
  double delta = ftk::solve_eigenvalues2x2(J, eig);
  // if(fabs(delta) < std::numeric_limits<double>::epsilon())
  if (delta >= 0) { // two real roots
    if (eig[0].real() * eig[1].real() < 0) {
      cp_type = SADDLE;
    } else if (eig[0].real() < 0) {
      cp_type = ATTRACTING;
    }
    else if (eig[0].real() > 0){
      cp_type = REPELLING;
    }
    else cp_type = SINGULAR;
  } else { // two conjugate roots
    if (eig[0].real() < 0) {
      cp_type = ATTRACTING_FOCUS;
    } else if (eig[0].real() > 0) {
      cp_type = REPELLING_FOCUS;
    } else 
      cp_type = CENTER;
  }
  return cp_type;
}

void extract_critical_points()
{
  fprintf(stderr, "extracting critical points...\n");
  int id = 0;
  for (int j = 1; j < DH-2; j ++) // avoiding borders..
    for (int i = 1; i < DW-2; i ++) {
      double u[4] = {grad(0, i, j), grad(0, i+1, j), grad(0, i+1, j+1), grad(0, i, j+1)};
      double v[4] = {grad(1, i, j), grad(1, i+1, j), grad(1, i+1, j+1), grad(1, i, j+1)};
      double mu[2][2];
      double X[2][2];

      int nroots = ftk::inverse_bilinear_interpolation(u, v, mu);
      for (int k = 0; k < nroots; k ++){
        // ftk::bilinear_interpolation3_location(X, mu, x);
        X[k][0] = i + mu[k][0], X[k][1] = j + mu[k][1]; 
        critical_point_t cp(mu[k], X[k], 0, 0);
        cp.simplex_id = id;
        cp.type = bilinear_critical_point_type(u, v, mu[k]);
        critical_points.insert(std::make_pair(2*id + k, cp));      
      }
      // if((fabs(mu[k][0] - 0.400304) < 1e-6) && (fabs(mu[k][1] - 0.505862) < 1e-6)){
      // if((j==1962) && (i==933)){
      //   std::cout << j << " " << i << ": number of roots = " << nroots << std::endl;
      //   for(int i=0; i<nroots; i++){
      //     std::cout << "root " << i << ": " << mu[i][0] << " " << mu[i][1] << std::endl;
      //   }
      //   for(int i=0; i<4; i++){
      //     std::cout << u[i] << " ";
      //   }
      //   std::cout << std::endl;
      //   for(int i=0; i<4; i++){
      //     std::cout << v[i] << " ";
      //   }
      //   std::cout << std::endl;
      // }

      // for (int k = 0; k < nroots; k ++)
      //   fprintf(stderr, "i=%d, j=%d, x=%f, y=%f\n", i, j, i + mu[k][0], j + mu[k][1]);
      id ++;
    }
}

void refill_gradient(int id, const float* grad_tmp){
  const float * grad_tmp_pos = grad_tmp;
  for (int i = 0; i < DH; i ++) {
    for (int j = 0; j < DW; j ++) {
      grad(id, j, i) = *(grad_tmp_pos ++);
    }
  }
}

std::unordered_map<int, critical_point_t> get_critical_points(){
  critical_points.clear();
  // derive_gradients();
  extract_critical_points();
  std::unordered_map<int, critical_point_t> result(critical_points);
  return result;
}

template<typename Type>
Type * readfile(const char * file, size_t& num){
  std::ifstream fin(file, std::ios::binary);
  if(!fin){
        std::cout << " Error, Couldn't find the file" << "\n";
        return 0;
    }
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(Type);
    fin.seekg(0, std::ios::beg);
    Type * data = (Type *) malloc(num_elements*sizeof(Type));
  fin.read(reinterpret_cast<char*>(&data[0]), num_elements*sizeof(Type));
  fin.close();
  num = num_elements;
  return data;
}

template<typename Type>
void writefile(const char * file, Type * data, size_t num_elements){
  std::ofstream fout(file, std::ios::binary);
  fout.write(reinterpret_cast<const char*>(&data[0]), num_elements*sizeof(Type));
  fout.close();
}

inline bool file_exists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}

std::unordered_map<int, critical_point_t> read_criticalpoints(const std::string& prefix){
  std::unordered_map<int, critical_point_t> cps;
  size_t num = 0;
  double * positions = readfile<double>((prefix + "_pos.dat").c_str(), num);
  int * type = readfile<int>((prefix + "_type.dat").c_str(), num);
  size_t * sid = readfile<size_t>((prefix + "_sid.dat").c_str(), num);
  printf("Read %ld critical points\n", num);
  for(int i=0; i<num; i++){
    critical_point_t p;
    p.x[0] = positions[2*i]; p.x[1] = positions[2*i+1];
    p.type = type[i];
    p.simplex_id = sid[i]; 
    cps.insert(std::make_pair(sid[i], p));
  }
  return cps;
}

void record_criticalpoints(const std::string& prefix, const std::vector<critical_point_t>& cps, bool write_sid=false){
  double * positions = (double *) malloc(cps.size()*2*sizeof(double));
  int * type = (int *) malloc(cps.size()*sizeof(int));
  int i = 0;
  for(const auto& cp:cps){
    positions[2*i] = cp.x[0];
    positions[2*i+1] = cp.x[1];
    type[i] = cp.type;
    i ++;
  }
  writefile((prefix + "_pos.dat").c_str(), positions, cps.size()*2);
  writefile((prefix + "_type.dat").c_str(), type, cps.size());
  if(write_sid){
    size_t * sid = (size_t *) malloc(cps.size()*sizeof(size_t));
    int i = 0;
    for(const auto& cp:cps){
      sid[i ++] = cp.simplex_id;
    }
    writefile((prefix + "_sid.dat").c_str(), sid, cps.size());
    free(sid);
  }
  free(positions);
  free(type);
}

int main(int argc, char **argv){
  // size_t starts[3] = {0, 0, 0}, 
  //        sizes[3]  = {1, size_t(DH), size_t(DW)};
  // DH = atoi(argv[1]);
  // DW = atoi(argv[2]);
  DW = atoi(argv[1]);
  DH = atoi(argv[2]);
  size_t num = 0;
  float * u = readfile<float>(argv[3], num);
  float * v = readfile<float>(argv[4], num);

  grad.reshape({2, static_cast<unsigned long>(DW), static_cast<unsigned long>(DH)});
  refill_gradient(0, u);
  refill_gradient(1, v);
  free(u);
  free(v);

  // auto critical_points_0 = get_critical_points();
  bool cp_file = file_exists("origin_M_sid.dat");
  cp_file ? printf("Critical point file found!\n") : printf("Critical point Not found, recomputing\n");
  auto critical_points_0 = cp_file ? read_criticalpoints("origin_M") : get_critical_points();
  std::cout << "critical_points number = " << critical_points_0.size() << std::endl;

  {
    std::cout << "read decompressed data\n";
    std::string fn_u = std::string(argv[3]) + ".out";
    std::string fn_v = std::string(argv[4]) + ".out";
    u = readfile<float>(fn_u.c_str(), num);
    v = readfile<float>(fn_v.c_str(), num);
    std::cout << "read " << num << " decompressed data done\n";
  }
  refill_gradient(0, u);
  refill_gradient(1, v);
  free(u);
  free(v);
  std::cout << "refill gradients done\n";
  auto critical_points_1 = get_critical_points();
  std::cout << "decompressed critical_points number = " << critical_points_1.size() << std::endl;
  int matches = 0;
  std::vector<critical_point_t> fp, fn, ft, m;
  std::vector<critical_point_t> origin;
  // int cp_type_change = 0;
  for(const auto& p:critical_points_0){
    auto cp = p.second;
    origin.push_back(cp);
    if(critical_points_1.find(p.first) != critical_points_1.end()){
      // matches ++;
      auto cp_1 = critical_points_1[p.first];
      if(cp.type != cp_1.type){
        std::cout << "Change from " << get_critical_point_type_string(cp.type)
          << " to " << get_critical_point_type_string(cp_1.type) <<
           ", positions from " << cp.x[0] << ", " << cp.x[1] << " to " << cp_1.x[0] << ", " << cp_1.x[1] << std::endl;
        // cp_type_change ++;
        ft.push_back(cp_1);
      }
      else m.push_back(cp_1);
    }
    else fn.push_back(p.second);
    // std::cout << std::endl;
  }
  for(const auto& p:critical_points_1){
    if(critical_points_0.find(p.first) == critical_points_0.end()){
      fp.push_back(p.second);
    }
  }
  std::cout << "FN number = " << fn.size() << std::endl; 
  for(const auto& cp:fn){
    // std::cout << "mu = " << cp.mu[0] << ", " << cp.mu[1] << ", " << cp.mu[2] << "; x = " << cp.x[0] << ", " << cp.x[1] << "; v = " << cp.v << std::endl;
    std::cout << "x = " << cp.x[0] << ", " << cp.x[1] << "; type = " << get_critical_point_type_string(cp.type) << std::endl;
  }
  std::cout << "FP number = " << fp.size() << std::endl; 
  for(const auto& cp:fp){
    // std::cout << "mu = " << cp.mu[0] << ", " << cp.mu[1] << ", " << cp.mu[2] << "; x = " << cp.x[0] << ", " << cp.x[1] << "; v = " << cp.v << std::endl;
    std::cout << "x = " << cp.x[0] << ", " << cp.x[1] << "; type = " << get_critical_point_type_string(cp.type) << std::endl;
  }
  std::cout << "FT number = " << ft.size() << std::endl; 
  for(const auto& cp:ft){
    std::cout << "x = " << cp.x[0] << ", " << cp.x[1] << "; type = " << get_critical_point_type_string(cp.type) << std::endl;
  }
  std::cout << "Ground truth = " << origin.size() << std::endl;
  std::cout << "TP = " << m.size() << std::endl;
  std::cout << "FP = " << fp.size() << std::endl;
  std::cout << "FN = " << fn.size() << std::endl;
  std::cout << "FT = " << ft.size() << std::endl;

  return 0;
}
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <hypermesh/conv.hh>
#include <hypermesh/ndarray.hh>
#include <ftk/numeric/rand.hh>

class conv_test : public testing::Test {
public:
  const int nruns = 100000;
  const double epsilon = 1e-9;
};

TEST_F(conv_test, 2D_conv_test) {
  hypermesh::ndarray<double> k({2,2}),d({3,3}),r({2,2});
  std::vector<double> kernel = {-1,2,1,2,},
                      data = {1,2,3,4,5,6,7,8,9,},
                      ans = {17,21,29,33,},
                      res;
  k.from_vector(kernel);
  d.from_vector(data);
  r = hypermesh::conv2D(d,k);
  r.to_vector(res);
  for (int i=0;i<4;++i){
    ans[i]/=4;
  }
  EXPECT_EQ(res,ans);
}

TEST_F(conv_test, 2D_conv_padding_test) {
  hypermesh::ndarray<double> k({3,3}),d({3,3}),r({3,3});
  std::vector<double> kernel = {1,2,1,0,0,0,-1,-2,-1,},
                      data = {1,2,3,4,5,6,7,8,9,},
                      ans = {-13,-20,-17,-18,-24,-18,13,20,17,},
                      res;
  k.from_vector(kernel);
  d.from_vector(data);
  r = hypermesh::conv2D(d,k,1);
  r.to_vector(res);
  for (int i=0;i<9;++i){
    ans[i]/=9;
  }
  EXPECT_EQ(res,ans);
}

TEST_F(conv_test, 2D_conv_gaussian_test) {
  int ksizex = 2, ksizey = 2;
  double sigma = 1.;
  hypermesh::ndarray<double> d({3,3}),r({2,2});
  d.from_vector({1,2,3,4,5,6,7,8,9,});
  std::vector<double> ans={3/4.,1.,6/4.,7/4.,},res;
  r = hypermesh::conv2D_gaussian(d,sigma,ksizex,ksizey);
  r.to_vector(res);
  EXPECT_EQ(res,ans);
}

TEST_F(conv_test, 3D_conv_test) {
  hypermesh::ndarray<double> d({3,3,3}),k({2,2,2}),r({2,2,2});
  std::vector<double> kernel = {1,2,1,0,0,0,-1,-2,},
                      ans = {-4, -3.875, -3.625, -3.5, -2.875, -2.75, -2.5, -2.375,},
                      data,
                      res;
  for (int i=0;i<27;i++){
    data.push_back(i+1);
  }
  k.from_vector(kernel);
  d.from_vector(data);
  r = hypermesh::conv3D(d,k,0);
  r.to_vector(res);
  EXPECT_EQ(res,ans);
}
TEST_F(conv_test, 3D_conv_padding_test) {
  hypermesh::ndarray<double> d({2,2,2}),k({2,2,2}),r;
  std::vector<double> kernel = {1,2,1,0,0,0,-1,-2,},
                      ans = { -0.25,-0.625,-0.25,-0.75,-1.375,-0.5,0,0,0,-1.25,-2,-0.5,-1.5,-1.875,-0.25,0.75,1.375,0.5,0,0.625,0.75,1.25,3,1.75,1.75,2.875,1,},
                      data,
                      res;
  for (int i=0;i<8;i++){
    data.push_back(i+1);
  }
  k.from_vector(kernel);
  d.from_vector(data);
  r = hypermesh::conv3D(d,k,1);
  r.to_vector(res);
  EXPECT_EQ(res,ans);
}
TEST_F(conv_test, 3D_conv_gaussian_test) {
  std::vector<double> ans={-0.6875, -0.8125, -1.0625, -1.1875, -1.8125, -1.9375, -2.1875, -2.3125},
                      data,
                      res;
  hypermesh::ndarray<double> d({3,3,3}),r({2,2,2});
  for (int i=0;i<27;i++)
    data.push_back(-i+1);
  d.from_vector(data);
  r = hypermesh::conv3D_gaussian(d,1.,2,2,2);
  r.to_vector(res);
  EXPECT_EQ(res,ans);
}
#ifndef _FTK_PARALLEL_VECTOR_H
#define _FTK_PARALLEL_VECTOR_H

#include <ftk/numerics/eig.hh>

namespace ftk {

template <typename ValueType>
inline bool parallel_vector(const ValueType v[9], const ValueType w[9], ValueType lambda[3])
{
  ValueType invV[9], invW[9];
  const auto detV = invmat3(v, invV),
             detW = invmat3(v, invW);
  
  ValueType m[9];
  if (detW > detV) mulmat3(invW, V, m);
  else mulmat3(invV, W, m);

  bool found = false;
  std::complex<ValueType> eig[3], eigvec[3][3];
  eig3(m, eig, eigvec);

  for (int i=0; i<3; i++) {
    if (eig[i].imag() != 0) continue; // non-real eigenvalue
    ValueType l[3] = {eigvec[i][0].real(), eigvec[i][1].real(), eigvec[i][2].real()};
    if (l[0] < 0 || l[1] < 0) continue;
    const auto sum = l[0] + l[1] + l[2];
    lambda[0] = l[0] / sum;
    lambda[1] = l[1] / sum;
    lambda[2] = l[2] / sum;
    return true;
  }
}

}

#endif

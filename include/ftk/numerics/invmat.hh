#ifndef _FTK_INVMAT_H
#define _FTK_INVMAT_H

namespace ftk {

template <class ValueType>
inline ValueType invmat2(const ValueType m[9], ValueType inv[9]) // returns the determinant
{
  inv[0] = m[3];
  inv[1] = -m[1];
  inv[2] = -m[2];
  inv[3] = m[0];

  ValueType det = m[0]*m[3] - m[1]*m[2];
  ValueType invdet = ValueType(1) / det;
  for (int i = 0; i < 4; i++)
    inv[i] = inv[i] * invdet;

  return det;
}

template <class ValueType>
inline ValueType invmat3(const ValueType m[9], ValueType inv[9]) // returns the determinant
{
  inv[0] =   m[4]*m[8] - m[5]*m[7];
  inv[1] = - m[1]*m[8] + m[2]*m[7];
  inv[2] =   m[1]*m[5] - m[2]*m[4];
  inv[3] = - m[3]*m[8] + m[5]*m[6];
  inv[4] =   m[0]*m[8] - m[2]*m[6];
  inv[5] = - m[0]*m[5] + m[2]*m[3];
  inv[6] =   m[3]*m[7] - m[4]*m[6];
  inv[7] = - m[0]*m[7] + m[1]*m[6];
  inv[8] =   m[0]*m[4] - m[1]*m[3];
  
  ValueType det = m[0]*inv[0] + m[1]*inv[3] + m[2]*inv[6];
  ValueType invdet = ValueType(1) / det;

  for (int i=0; i<9; i++)
    inv[i] = inv[i] * invdet;

  return det;
}

template <class ValueType>
inline ValueType invmat4(const ValueType m[16], ValueType inv[16]) // returns det
{
  inv[0] =   m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15]
           + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
  inv[4] =  -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15]
           - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
  inv[8] =   m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15]
           + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
  inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14]
           - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
  inv[1] =  -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15]
           - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
  inv[5] =   m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15]
           + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
  inv[9] =  -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15]
           - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
  inv[13] =  m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14]
           + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
  inv[2] =   m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15]
           + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
  inv[6] =  -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15]
           - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
  inv[10] =  m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15]
           + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
  inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14]
           - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
  inv[3] =  -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11]
           - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
  inv[7] =   m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11]
           + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
  inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11]
           - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
  inv[15] =  m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10]
           + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

  ValueType det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12];
  ValueType invdet = ValueType(1) / det;

  for (int i = 0; i < 16; i++)
    inv[i] = inv[i] * invdet;

  return det;
}

}

#endif

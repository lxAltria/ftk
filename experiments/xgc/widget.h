#ifndef _WIDGET_H
#define _WIDGET_H

#include <GL/glew.h>
#include <QGLWidget>
#include <QList>
#include <QVector>
#include <QVector3D>
#include <QMatrix4x4>
#include <cmath>
#include <set>
#include <ftk/ndarray.hh>
#include <ftk/mesh/simplex_2d_mesh.hh>
#include "trackball.h"

class QMouseEvent;
class QKeyEvent; 
class QWheelEvent; 

/* 
 * \class   CGLWidget
 * \author  Hanqi Guo
 * \brief   A light-weight Qt-based vortex viewer
*/
class CGLWidget : public QGLWidget
{
  // Q_OBJECT

public:
  CGLWidget(const QGLFormat& fmt=QGLFormat::defaultFormat(), QWidget *parent=NULL, QGLWidget *sharedWidget=NULL); 
  ~CGLWidget(); 

  void set_mesh(const ftk::simplex_2d_mesh<>& m);

  // void loadData(const std::string& path);
  // void loadDataH5(const std::string& path);

protected:
  void initializeGL(); 
  void resizeGL(int w, int h); 
  void paintGL();

  void mousePressEvent(QMouseEvent*); 
  void mouseMoveEvent(QMouseEvent*);
  void keyPressEvent(QKeyEvent*); 
  void wheelEvent(QWheelEvent*); 

  int advanceSlice();
  int recedeSlice();
  int advanceTimestep();
  int recedeTimestep();

  void updateDataGL();

private:
  int currentTimestep = 120;
  int currentSlice = 0;

private:
  CGLTrackball _trackball;
  QMatrix4x4 _projmatrix, _mvmatrix; 

private: // camera
  const float _fovy, _znear, _zfar; 
  const QVector3D _eye, _center, _up;

  GLuint tex;

private:
  const int nt = 1; // for now
  
  std::vector<float> f_vertices;
  std::vector<float> f_colors, f_label_colors;
}; 

#endif
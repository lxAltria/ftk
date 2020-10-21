#ifndef _FTK_FEATURE_SURFACE_HH
#define _FTK_FEATURE_SURFACE_HH

#include <ftk/filters/feature_curve.hh>

#if FTK_HAVE_VTK
#include <vtkDoubleArray.h>
#include <vtkUnsignedIntArray.h>
#include <vtkPolyData.h>
#include <vtkTriangle.h>
#include <vtkQuad.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLPolyDataWriter.h>
#endif

namespace ftk {

struct feature_surface_t {
  std::vector<feature_point_t> pts;
  std::vector<std::array<int, 3>> tris; 
  std::vector<std::array<int, 4>> quads;

  void triangulate(); // WIP
  void reorient(); // WIP

#if FTK_HAVE_VTK
  vtkSmartPointer<vtkUnstructuredGrid> to_vtu() const; // preferred
  vtkSmartPointer<vtkPolyData> to_vtp() const;
#endif
};

inline feature_surface_t::triangulate()
{
  // 1. find all edges of triangles
  // 2. for each quad, find all combinations of edges
  //    - there should be at most four edges identified
  //    - TODO
}

#if FTK_HAVE_VTK
inline vtkSmartPointer<vtkPolyData> feature_surface_t::to_vtp() const
{
  vtkSmartPointer<vtkPolyData> poly = vtkPolyData::New();
  vtkSmartPointer<vtkPoints> points = vtkPoints::New();
  vtkSmartPointer<vtkCellArray> cells = vtkCellArray::New();

  vtkSmartPointer<vtkDataArray> array_id = vtkUnsignedIntArray::New();
  array_id->SetName("id");
  array_id->SetNumberOfComponents(1);
  array_id->SetNumberOfTuples(pts.size());

  vtkSmartPointer<vtkDataArray> array_time = vtkDoubleArray::New();
  array_time->SetName("time");
  array_time->SetNumberOfComponents(1);
  array_time->SetNumberOfTuples(pts.size());

  vtkSmartPointer<vtkDataArray> array_grad = vtkDoubleArray::New();
  array_grad->SetNumberOfComponents(3);
  array_grad->SetNumberOfTuples(pts.size());

  for (int i = 0; i < pts.size(); i ++) {
    const auto &p = pts[i];
    points->InsertNextPoint(p.x[0], p.x[1], p.x[2]);
    array_id->SetTuple1(i, p.id);
    array_time->SetTuple1(i, p.t);
    array_grad->SetTuple3(i, p.v[0], p.v[1], p.v[2]);
  }
  poly->SetPoints(points);
  poly->GetPointData()->AddArray(array_id);
  poly->GetPointData()->AddArray(array_time);
  poly->GetPointData()->SetNormals(array_grad);

  for (int i = 0; i < tris.size(); i ++) {
    const auto &c = tris[i];
    vtkSmartPointer<vtkTriangle> tri = vtkTriangle::New();
    for (int j = 0; j < 3; j ++)
      tri->GetPointIds()->SetId(j, c[j]);
    cells->InsertNextCell(tri);
  }
  for (int i = 0; i < quads.size(); i ++) {
    const auto &q = quads[i];
    vtkSmartPointer<vtkQuad> quad = vtkQuad::New();
    for (int j = 0; j < 4; j ++)
      quad->GetPointIds()->SetId(j, q[j]);
    cells->InsertNextCell(quad);
  }
  poly->SetPolys(cells);

  return poly;
}

inline vtkSmartPointer<vtkUnstructuredGrid> feature_surface_t::to_vtu() const
{
  vtkSmartPointer<vtkUnstructuredGrid> grid = vtkUnstructuredGrid::New();
  vtkSmartPointer<vtkPoints> points = vtkPoints::New();

  vtkSmartPointer<vtkDataArray> array_id = vtkUnsignedIntArray::New();
  array_id->SetName("id");
  array_id->SetNumberOfComponents(1);
  array_id->SetNumberOfTuples(pts.size());

  vtkSmartPointer<vtkDataArray> array_time = vtkDoubleArray::New();
  array_time->SetName("time");
  array_time->SetNumberOfComponents(1);
  array_time->SetNumberOfTuples(pts.size());
  
  vtkSmartPointer<vtkDataArray> array_scalar = vtkDoubleArray::New();
  array_scalar->SetName("scalar");
  array_scalar->SetNumberOfComponents(1);
  array_scalar->SetNumberOfTuples(pts.size());
  
  vtkSmartPointer<vtkDataArray> array_grad = vtkDoubleArray::New();
  array_grad->SetNumberOfComponents(3);
  array_grad->SetNumberOfTuples(pts.size());

  for (int i = 0; i < pts.size(); i ++) {
    const auto &p = pts[i];
    points->InsertNextPoint(p.x[0], p.x[1], p.x[2]);
    array_id->SetTuple1(i, p.id);
    array_time->SetTuple1(i, p.t);
    array_scalar->SetTuple1(i, p.scalar[0]);
    array_grad->SetTuple3(i, p.v[0], p.v[1], p.v[2]);
  }
  grid->SetPoints(points);
  grid->GetPointData()->AddArray(array_id);
  grid->GetPointData()->AddArray(array_time);
  grid->GetPointData()->AddArray(array_scalar);
  grid->GetPointData()->SetNormals(array_grad);

  for (int i = 0; i < tris.size(); i ++) {
    const auto &c = tris[i];
    vtkIdType ids[3] = {c[0], c[1], c[2]};
    grid->InsertNextCell(VTK_TRIANGLE, 3, ids);
  }

  return grid;
}
#endif

inline void feature_surface_t::reorient()
{
  fprintf(stderr, "reorienting, #pts=%zu, #tris=%zu\n", pts.size(), tris.size());
  auto edge = [](int i, int j) {
    if (i > j) std::swap(i, j);
    return std::make_tuple(i, j);
  };

  // 1. build triangle-triangle graph
  std::map<std::tuple<int, int>, std::set<int>> edge_triangle;
  for (int i = 0; i < tris.size(); i ++) {
    auto tri = tris[i];
    for (int j = 0; j < 3; j ++)
      edge_triangle[edge(tri[j], tri[(j+1)%3])].insert(i);
  }

  auto chirality = [](std::array<int, 3> a, std::array<int, 3> b) {
    std::vector<std::tuple<int, int>> ea, eb;
    for (int i = 0; i < 3; i ++) {
      ea.push_back(std::make_tuple(a[i], a[(i+1)%3]));
      eb.push_back(std::make_tuple(b[i], b[(i+1)%3]));
    }

    for (int i = 0; i < 3; i ++)
      for (int j = 0; j < 3; j ++) {
        if (ea[i] == eb[i]) return 1;
        else if (ea[i] == std::make_tuple(std::get<1>(eb[j]), std::get<0>(eb[j]))) 
          return -1;
      }

    assert(false);
    return 0;
  };

  // 2. reorientate triangles with bfs
  std::set<int> visited;
  std::queue<int> Q;
  Q.push(0);
  visited.insert(0);

  while (!Q.empty()) {
    auto current = Q.front();
    Q.pop();

    const int i = current;

    for (int j = 0; j < 3; j ++) {
      auto e = edge(tris[i][j], tris[i][(j+1)%3]);
      auto neighbors = edge_triangle[e];
      neighbors.erase(i);

      // fprintf(stderr, "#neighbors=%zu\n", neighbors.size());
      for (auto k : neighbors)
        if (visited.find(k) == visited.end()) {
          // fprintf(stderr, "pushing %d, chi=%d\n", k, chirality(tris[i], tris[k]));
          if (chirality(tris[i], tris[k]) < 0) {
            // fprintf(stderr, "flipping %d\n", k);
            std::swap(tris[k][0], tris[k][1]);
          }
          Q.push(k); // std::make_tuple(k, chirality(tris[i], tris[k])));
          visited.insert(k);
        }
    }
  }
}

}

#endif

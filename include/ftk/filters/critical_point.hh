#ifndef _FTK_CRITICAL_POINT_T_HH
#define _FTK_CRITICAL_POINT_T_HH

#include <ftk/ftk_config.hh>
#include <ftk/external/diy/serialization.hpp>
#include <ftk/external/json.hh>

namespace ftk {

using nlohmann::json;

// template <int N/*dimensionality*/, typename ValueType=double, typename IntegerType=unsigned long long>
struct critical_point_t {
  double operator[](size_t i) const {return x[i];}
  double x[3] = {0}; // coordinates 
  double t = 0.0; // time
  int timestep = 0; 
  // double rx[4] = {0}; // coordinates in transformed (e.g. curvilinear) grid, if eligible
  double scalar[FTK_CP_MAX_NUM_VARS] = {0};
  unsigned int type = 0;
  bool ordinal = false;
  unsigned long long tag = 0, id = 0;

  // constexpr size_t size() const noexcept { return sizeof(critical_point_t<N, ValueType, IntegerType>); }
  constexpr size_t size() const noexcept { return sizeof(critical_point_t); }

  std::ostream& print(std::ostream& os, const int cpdims, const std::vector<std::string>& scalar_components) const {
    if (cpdims == 2) os << "x=(" << x[0] << ", " << x[1] << "), ";
    else os << "x=(" << x[0] << ", " << x[1] << ", " << x[2] << "), ";
    os << "t=" << t << ", ";

    for (int k = 0; k < scalar_components.size(); k ++)
      os << scalar_components[k] << "=" << scalar[k] << ", ";
    
    os << "type=" << critical_point_type_to_string(cpdims, type, scalar_components.size()) << ", "; 
    os << "timestep=" << timestep << ", ";
    os << "ordinal=" << ordinal << ", ";
    os << "tag=" << tag << ", "; 
    os << "id=" << id;  // << std::endl;
    return os;
  }
};

struct critical_point_traj_t : public std::vector<critical_point_t>
{
  unsigned long long identifier;
  bool complete = false, loop = false;
  std::array<double, FTK_CP_MAX_NUM_VARS> max, min, persistence;
  std::array<double, 3> bbmin, bbmax; // bounding box
  double tmin, tmax; // time bounding box
  unsigned int consistent_type = 0; // 0 if no consistent type

  void discard_interval_points() {
    critical_point_traj_t traj;
    traj.identifier = identifier;
    traj.loop = loop;
    for (auto i = 0; i < size(); i ++) {
      if (at(i).ordinal) 
        traj.push_back(at(i));
    }
    traj.update_statistics();
    *this = traj;
  }
  
  void discard_degenerate_points() {
    critical_point_traj_t traj;
    traj.identifier = identifier;
    traj.loop = loop;
    for (auto i = 0; i < size(); i ++) {
      if (at(i).type != 1 && at(i).type != 0) // degenerate or unknown
        traj.push_back(at(i));
    }
    traj.update_statistics();
    *this = traj;
  }

  void update_statistics() {
    if (empty()) return; // nothing to do

    max.fill( std::numeric_limits<double>::lowest() );
    min.fill( std::numeric_limits<double>::max() );
    bbmax.fill( std::numeric_limits<double>::lowest() );
    bbmin.fill( std::numeric_limits<double>::max() );
    tmax = std::numeric_limits<double>::lowest();
    tmin = std::numeric_limits<double>::max();

    for (auto i = 0; i < size(); i ++) {
      for (int k = 0; k < FTK_CP_MAX_NUM_VARS; k ++) {
        max[k] = std::max(max[k], at(i).scalar[k]);
        min[k] = std::min(min[k], at(i).scalar[k]);
      }
      for (int k = 0; k < 3; k ++) {
        bbmax[k] = std::max(bbmax[k], at(i).x[k]);
        bbmin[k] = std::min(bbmin[k], at(i).x[k]);
      }
      tmax = std::max(tmax, at(i).t);
      tmin = std::min(tmin, at(i).t);
    }

    for (int k = 0; k < FTK_CP_MAX_NUM_VARS; k ++)
      persistence[k] = max[k] - min[k];

    consistent_type = at(0).type;
    for (auto i = 0; i < size(); i ++)
      if (consistent_type != at(i).type) {
        consistent_type = 0;
        break;
      }
  }
 
  std::vector<critical_point_traj_t> to_consistent_sub_traj() const {
    std::vector<critical_point_traj_t> results;
    critical_point_traj_t subtraj;
    unsigned int current_type;

    for (auto i = 0; i < size(); i ++) {
      if (subtraj.empty())
        current_type = at(i).type;

      if (at(i).type == current_type)
        subtraj.push_back(at(i));
      else {
        subtraj.update_statistics();
        results.push_back(subtraj);
        subtraj.clear();
      }
    }

    return results;
  }
};

}

// serialization w/ json
namespace nlohmann
{
  using namespace ftk;

  template <>
  struct adl_serializer<critical_point_t> {
    static void to_json(json &j, const critical_point_t& cp) {
      j["x"] = {cp.x[0], cp.x[1], cp.x[2]};
      j["t"] = cp.t;
      j["timestep"] = cp.timestep;
      j["scalar"] = std::vector<double>(cp.scalar, cp.scalar+FTK_CP_MAX_NUM_VARS);
      j["type"] = cp.type;
      j["ordinal"] = cp.ordinal;
      j["tag"] = cp.tag;
      j["id"] = cp.id;
    }

    static void from_json(const json& j,critical_point_t& cp) {
      for (int i = 0; i < 3; i ++)
        cp.x[i] = j["x"][i];
      cp.t = j["t"];
      cp.timestep = j["timestep"];
      for (int i = 0; i < FTK_CP_MAX_NUM_VARS; i ++)
        cp.scalar[i] = j["scalar"][i];
      cp.type = j["type"];
      cp.ordinal = j["ordinal"];
      cp.tag = j["tag"];
      cp.id = j["id"];
    }
  };
  
  template <>
  struct adl_serializer<critical_point_traj_t> {
    static void to_json(json &j, const critical_point_traj_t& t) {
      j = {
        {"id", t.identifier},
        {"max", t.max},
        {"min", t.min},
        {"persistence", t.persistence},
        {"bbmin", t.bbmin},
        {"bbmax", t.bbmax},
        {"tmin", t.tmin},
        {"tmax", t.tmax},
        {"consistent_type", t.consistent_type},
        {"traj", static_cast<std::vector<critical_point_t>>(t)}
      };
    }

    static void from_json(const json&j, critical_point_traj_t& t) {
      t.identifier = j["id"];
      t.max = j["max"];
      t.min = j["min"];
      t.persistence = j["persistence"];
      t.bbmin = j["bbmin"];
      t.bbmax = j["bbmax"];
      t.tmin = j["tmin"];
      t.tmax = j["tmax"];
      t.consistent_type = j["consistent_type"];
      std::vector<critical_point_t> traj = j["traj"];
      t.clear();
      t.insert(t.begin(), traj.begin(), traj.end());
    }
  };
}



// serialization
namespace diy {
  // template <int N, typename V, typename I> 
  // static void save(diy::BinaryBuffer& bb, const ftk::critical_point_t<N, V, I> &cp) {
  static void save(diy::BinaryBuffer& bb, const ftk::critical_point_t &cp) {
    for (int i = 0; i < 3; i ++)
      diy::save(bb, cp.x[i]);
    diy::save(bb, cp.t);
    for (int i = 0; i < FTK_CP_MAX_NUM_VARS; i ++)
      diy::save(bb, cp.scalar[i]);
    diy::save(bb, cp.type);
    diy::save(bb, cp.ordinal);
    diy::save(bb, cp.tag);
    diy::save(bb, cp.id);
  }

  // template <int N, typename V, typename I> 
  // static void load(diy::BinaryBuffer& bb, ftk::critical_point_t<N, V, I> &cp) {
  static void load(diy::BinaryBuffer& bb, ftk::critical_point_t &cp) {
    for (int i = 0; i < 4; i ++) 
      diy::load(bb, cp.x[i]);
    diy::save(bb, cp.t);
    for (int i = 0; i < FTK_CP_MAX_NUM_VARS; i ++)
      diy::load(bb, cp.scalar[i]);
    diy::load(bb, cp.type);
    diy::load(bb, cp.ordinal);
    diy::load(bb, cp.tag);
    diy::load(bb, cp.id);
  }
  
  static void save(diy::BinaryBuffer& bb, const ftk::critical_point_traj_t &t) {
    diy::save(bb, t.complete);
    diy::save(bb, t.max);
    diy::save(bb, t.min);
    diy::save(bb, t.persistence);
    diy::save(bb, t.bbmin);
    diy::save(bb, t.bbmax);
    diy::save(bb, t.tmin);
    diy::save(bb, t.tmax);
    diy::save(bb, t.consistent_type);
    diy::save<std::vector<ftk::critical_point_t>>(bb, t);
  }
  
  static void load(diy::BinaryBuffer& bb, ftk::critical_point_traj_t &t) {
    diy::load(bb, t.complete);
    diy::load(bb, t.max);
    diy::load(bb, t.min);
    diy::load(bb, t.persistence);
    diy::load(bb, t.bbmin);
    diy::load(bb, t.bbmax);
    diy::load(bb, t.tmin);
    diy::load(bb, t.tmax);
    diy::load(bb, t.consistent_type);
    diy::load<std::vector<ftk::critical_point_t>>(bb, t);
  }
}

#endif

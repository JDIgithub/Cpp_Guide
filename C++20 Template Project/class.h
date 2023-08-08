#include <cstring>

class Point{

public:
  Point();
  ~Point();
  double length() const;

private:
  double m_x;
  double m_y;

public:

  inline static size_t m_point_count {};  // We can initialize inline static member "in class"


};


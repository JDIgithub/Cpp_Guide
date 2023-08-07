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

  static size_t m_point_count;  // We can not initialize static member "in class"
  // because it is not tied to the class

};


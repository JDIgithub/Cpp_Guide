#include <cstring>
#include <iostream>
#include <stdint.h>
#include <vector>



class Point{

public:
  Point();
  Point(double x, double y);
  ~Point();
  double length() const;


  // But they can use member variables of some object that is passed to them of course
  static void print_point_info(const Point& p){
    std::cout << "Point: x = " << p.m_x << " y = " << p.m_y << "]" << std::endl;
  }

  // It can also use static member variables because they also belong to the blueprint
  static size_t get_point_count(){
    return m_point_count;
  }

private:
  double m_x;
  double m_y;

public:

  inline static size_t m_point_count {};  // We can initialize inline static member "in class"


};

/*

class Outer{

public:
  Outer(int int_param, double double_param);
  Outer();
  ~Outer();
  void createInnerClassObject();

private:
  int m_var1;
  int m_var2;
  inline static int static_int{45};

  class Inner{  // In private section -> private class
    public:
      Inner(double double_param);
    private:
      double inner_var;
  };

};*/
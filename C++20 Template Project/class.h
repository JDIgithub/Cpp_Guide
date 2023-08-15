
#ifndef CLASS_H
#define CLASS_H

#include <cstring>
#include <iostream>
#include <stdint.h>
#include <vector>
#include <utility>
//#include "my_utility.h"

// Header
class Point
{
	friend std::ostream& operator<< (std::ostream& out , const Point& p);
    
  friend  bool operator== (const Point& left , const Point& right);
  friend  bool operator< (const Point& left , const Point& right);
   
public:
	Point() = default;
	Point(double x, double y) : 
		m_x(x), m_y(y){
	}
	~Point() = default;

private: 
	double length() const;   // Function to calculate distance from the point(0,0) 
	double m_x{};
	double m_y{};
};

inline std::ostream& operator<< (std::ostream& out , const Point& p){
	out << "Point [ x : " << p.m_x << ", y : " << p.m_y << 
        " length : " << p.length() <<  "]" ;
	return out;
}
inline   bool operator== (const Point& left , const Point& right) {
     return (left.length() == right.length());
 }
inline   bool operator< (const Point& left , const Point& right) {
     return (left.length() < right.length());
 }




 //using namespace std::rel_ops;
 



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



#endif // CLASS_H
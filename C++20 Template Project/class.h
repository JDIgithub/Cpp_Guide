
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


// person.h (Base class) ----------------------------------------------------
class Person
{
    friend std::ostream& operator<<(std::ostream& , const Person& person);
public:
    Person() ;
    Person(std::string_view fullname,int age,
    std::string_view address);
    Person(const Person& source);
    ~Person();
    
    //Getters
    std::string get_full_name()const{
        return m_full_name;
    }
    
    int get_age()const{
        return m_age;
    }
    
    std::string get_address()const{
        return m_address;
    }


    int add(int a, int b) const{
        return a + b ;
    }

    int add(int a, int b, int  c) const{
        return a + b + c;
    }

    void do_something() const;
public:
    std::string m_full_name{"None"};
protected: 
    int m_age{0};
private : 
    std::string m_address{"None"};
};

// engineer.h
class Engineer : public Person
{
  using Person::Person; // Inheriting the Constructor of the base class
  friend std::ostream& operator<<(std::ostream& out , const Engineer& operand);
public:
  // Now we do not need so many constructor for engineer class because we inherit the base constructors
  // But the base constructors can not initialize any engineer members so if we want to do that we need engineers own constructors for that
  Engineer(const Engineer& source); // Also Copy constructors can not be inherited so we should have engineers own
  ~Engineer();
    
  void build_something(){
    m_full_name = "John Snow"; // OK
    m_age = 23; // OK
  }

  int get_contract_count() const{
    return contract_count;
  }
    
private : 
  int contract_count{0};
};

 // And it will make these getters protected 

// civilengineer.h ----------------------------------------------------------
class CivilEngineer : public Engineer
{
    friend std::ostream& operator<<(std::ostream&, const CivilEngineer& operand);
public:
    CivilEngineer();
    CivilEngineer(std::string_view fullname,int age,
    std::string_view address,int contract_count, std::string_view speciality);
    CivilEngineer(const CivilEngineer& source);
    ~CivilEngineer() ;
    
    void build_road(){
        //get_full_name(); // Compiler error
        ///m_full_name = "Daniel Gray"; //Compiler error
        //m_age = 45; // Compiler error

        add(10,2);
        add(10,2,4);
    }

    public : 
        //using Person::do_something; // Compiler error
	
private : 
    std::string m_speciality{"None"};

};



class Parent
{
public:
    Parent() = default;
    Parent(int member_var) : m_member_var(member_var){   
    }
    ~Parent() = default;
    
    void print_var()const{
        std::cout << "The value in parent is : " << m_member_var << std::endl;
    }
protected: 
    int m_member_var{100};
};

class Child : public Parent 
{
public:
    Child();
    Child( int member_var) : m_member_var(member_var){
    }
    ~Child() = default;
    
    void print_var()const{
        std::cout << "The value in child is : " << m_member_var << std::endl;
    }
    
    void show_values()const{
        std::cout << "The value in child is :" << m_member_var << std::endl;
        std::cout << "The value in parent is : " << Parent::m_member_var << std::endl;
                // The value in parent must be in accessible scope from the derived class.
    }
private: 
    int m_member_var{1000};
};




class Shape
{
public:
  Shape() = default;
  Shape(std::string_view description);
  ~Shape();
    
  virtual void draw() const{
    std::cout << "Shape::draw() called. Drawing " << m_description << std::endl;
  }
    
protected : 
  std::string m_description{""};
};

class Oval : public Shape
{
public:
  Oval()= default;
  Oval(double x_radius, double y_radius, std::string_view description);
  ~Oval();
    
  virtual void draw() const{
    std::cout << "Oval::draw() called. Drawing " << m_description <<
    " with m_x_radius : " << m_x_radius << " and m_y_radius : " << m_y_radius 
    << std::endl;
  }

protected:
  double get_x_rad() const{
    return m_x_radius;
  }
    
  double get_y_rad() const{
    return m_y_radius;
  }
 
private : 
  double m_x_radius{0.0};
  double m_y_radius{0.0};
};

class Circle : public Oval
{
public:
  Circle() = default;
  Circle(double radius,std::string_view description);
  ~Circle();
    
  virtual void draw() const{
    std::cout << "Circle::draw() called. Drawing " << m_description <<
    " with radius : " << get_x_rad() << std::endl;        
  }

};












#endif // CLASS_H
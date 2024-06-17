#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>
#include <list>
#include <random>
#include <atomic>
#include <csignal>
#include <optional>

using namespace std;


class Point{

public:
  int x;
  int * y;

  Point(): x(0){
    y = new int(0);
  };
  Point(int x, int y){
    this->x = x;
    this->y = new int(y);
  };
  Point(const Point& copied_Point){
    x = copied_Point.x;
    y = new int(*copied_Point.y);
  }
  // Point(const Point& copied_Point) = delete; -> non-copyable object  or make it private
  /* We can also use delegation
  Point(const Point& copied_Point): Point(copied_Point.x,*copied_Point.y){
  }
  */

  Point(Point&& moved_Point){
    x = moved_Point.x;
    y = moved_Point.y;  
    moved_Point.y = nullptr;
  }

  Point& operator=(const Point& copied_Point){
    if(this == &copied_Point){ return *this;}
    if(y == nullptr){ y = new int; }
    *y = *copied_Point.y;
    x = copied_Point.x;
    return *this;
  }

  Point& operator=(Point&& moved_Point){
    if(this == &moved_Point){ return *this;}
    delete y;  // Free existing resource
    x = moved_Point.x;
    y = moved_Point.y;
    moved_Point.y = nullptr;
    return *this;
  }


  ~Point(){
    delete y;
  }


};





int main() {

  Point p1(4,8);
  Point p2(p1);
  Point p3(1,2);

  p3 = p1;

  Point p4(std::move(p1));

  auto [a,b] = p4;

  

  int x = 4545;

  return 0;
}





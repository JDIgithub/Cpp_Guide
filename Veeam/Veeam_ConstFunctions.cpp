#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <stack>
#include <cassert>

#include <iostream>
#include <limits>


class A{

public:
  int f1() const {return _val;} 
  const int& f2(){return _val;}
  int f3(){return _val;}
private:
  int _val = 5;

};


void Do1(const A& obj) { int val = obj.f1(); }  // Right answer
void Do2(const A& obj) { int val = obj.f2(); }  // Compilation Error: f2() is not const-qualified
void Do3(const A& obj) { int val = obj.f3(); }  // Compilation Error: f3() is not const-qualified





int main() {

  A obj;
  const A& obj2 = obj;

  int val1 = obj.f1();
  int val2 = obj.f2();
  int val3 = obj.f3();

  int val4 = obj2.f1();
  int val5 = obj2.f2();
  int val6 = obj2.f3();


  return 0;
}


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
  ~A(){printf("~A()");}

};

class B : public A{

public:
  ~B(){printf("~B()");}

};

int main(){

  A* obj = new B;
  delete obj;
  return 0;

}


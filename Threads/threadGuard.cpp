#include <iostream>
#include <thread>

class thread_guard {
  
  std::thread thr;

public:
  // Constructor takes rvalue reference argument (std::thread is move-only)
  explicit thread_guard(std::thread&& thr): thr(std::move(thr)){}
  ~thread_guard(){
    if(thr.joinable()){
      thr.join();
    }
  }

  thread_guard(const thread_guard&) = delete; // Prevents copying
  thread_guard& operator=(const thread_guard&) = delete;

  // The move assignment operator is not synthesized

};

void hello(){
  std::cout << "Hello, Thread!\n";
}


// Main function
int main() {
  
  try{
    std::thread thr(hello);
    thread_guard tguard{std::move(thr)};
    throw std::exception();
  } catch ( std::exception& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  return 0;
}





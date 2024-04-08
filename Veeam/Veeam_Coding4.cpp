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
#include <cstring>
#include <climits>

/*
std::mutex mtx;
std::condition_variable cv;
int count = 1;

void PrintOdd() {
    for (int i = 0; i < N; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return count % 2 != 0; });
        std::cout << count++ << std::endl;
        cv.notify_one();
    }
}

void PrintEven() {
    for (int i = 0; i < N; ++i) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [] { return count % 2 == 0; });
        std::cout << count++ << std::endl;
        cv.notify_one();
    }
}

int main() {
    const int N = 4; // Number of iterations

    std::thread oddThread(PrintOdd);
    std::thread evenThread(PrintEven);

    oddThread.join();
    evenThread.join();

    return 0;
}

*/
void PrintNextOddNumber();
void PrintNextEvenNumber();
/*



std::thread oddThread([]{
for (int i = 0; i <N; ++i){
 PrintOdd();
}
});

std::thread evenThread([]{
for (int i = 0; i <N; ++i){
 PrintEven();
}
});

// They are implemented this way :

void PrintOdd() {
  PrintNextOddNumber();
}

void PrintEven(){
  PrintNextEvenNumber();
}
*/


int currentNumber = 1;
const int N = 4;  // Both thread will print 4-times to get 1-8
std::mutex mtx;
std::condition_variable cv;

void PrintOdd() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return currentNumber % 2 != 0; });  // Wait until it's odd
    std::cout << currentNumber << std::endl;
    ++currentNumber;
    lock.unlock();
    cv.notify_all();  // Notify even thread
}

void PrintEven() {
    std::unique_lock<std::mutex> lock(mtx);
    cv.wait(lock, [] { return currentNumber % 2 == 0; });  // Wait until it's even
    std::cout << currentNumber << std::endl;
    ++currentNumber;
    lock.unlock();
    cv.notify_all();  // Notify odd thread
}

int main() {
    std::thread oddThread([] {  for (int i = 0; i < N; ++i) { PrintOdd(); } });
    std::thread evenThread([] { for (int i = 0; i < N; ++i) { PrintEven(); } });
    oddThread.join();
    evenThread.join();
    return 0;
}



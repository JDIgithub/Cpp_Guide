Multithreading Questions:
Question 1:

cpp
Copy code
#include <iostream>
#include <thread>
#include <vector>

void printNumber(int num) {
    std::cout << num << ' ';
}

int main() {
    std::vector<std::thread> threads;
    for(int i = 0; i < 5; ++i) {
        threads.push_back(std::thread(printNumber, i));
    }
    for(auto& t : threads) {
        t.join();
    }
    return 0;
}
What is the expected output of the program?
Answer 1:

The output will be the numbers 0 to 4 printed in any order. Each number will be followed by a space. The exact order is non-deterministic due to the nature of thread scheduling by the operating system.
Question 2:

cpp
Copy code
#include <iostream>
#include <thread>
#include <mutex>

std::mutex coutMutex;

void printWithMutex(int num) {
    coutMutex.lock();
    std::cout << "Number: " << num << std::endl;
    coutMutex.unlock();
}

int main() {
    std::thread t1(printWithMutex, 1);
    std::thread t2(printWithMutex, 2);
    t1.join();
    t2.join();
    return 0;
}
What is the purpose of the coutMutex, and how does it affect the program's output?
Answer 2:

The coutMutex ensures that the threads do not interrupt each other while writing to std::cout. This means that the two lines of output will not be interleaved or mixed up. Each number will be printed with its accompanying text in a separate, atomic operation.
Question 3:

cpp
Copy code
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> counter(0);

void incrementCounter() {
    for (int i = 0; i < 100; ++i) {
        counter++;
    }
}

int main() {
    std::thread t1(incrementCounter);
    std::thread t2(incrementCounter);
    t1.join();
    t2.join();
    std::cout << "Final counter value: " << counter << std::endl;
    return 0;
}
What is the significance of std::atomic<int> in this code, and what will be the final value of counter?
Answer 3:

std::atomic<int> is used to ensure that increment operations on counter are atomic, meaning that each increment is completed as a single, indivisible operation. This prevents data races and ensures thread safety. The final value of counter will be 200 since each thread increments it 100 times.
Question 4:

cpp
Copy code
#include <iostream>
#include <thread>
#include <chrono>

void threadFunction() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Finished waiting" << std::endl;
}

int main() {
    std::thread t(threadFunction);
    t.detach();
    std::cout << "Main thread is free now" << std::endl;
    return 0;
}
Explain the behavior of detach() in this context, and what might be the potential output of the program?
Answer 4:

detach() is used to separate the thread t from the std::thread object, allowing t to continue execution independently. This allows the main thread to exit without waiting for t to finish. The potential output might have "Main thread is free now" printed before "Finished waiting", but "Finished waiting" may not be printed at all if the main program exits before the detached thread completes its execution.
Question 5:

cpp
Copy code
#include <iostream>
#include <thread>
#include <mutex>

std::mutex resourceMutex;
int resource = 0;

void accessResource() {
    std::lock_guard<std::mutex> lock(resourceMutex);
    resource++;
    std::cout << "Resource value: " << resource << std::endl;
}

int main() {
    std::thread t1(accessResource);
    std::thread t2(accessResource);
    t1.join();
    t2.join();
    return 0;
}
What does std::lock_guard do in this code, and in what order will the resource values be printed?
Answer 5:

std::lock_guard is a mutex wrapper that provides a convenient RAII-style mechanism for owning a mutex for the duration of a scoped block. When accessResource is called by two threads, std::lock_guard ensures that only one thread can access and modify the resource at a time. It locks the mutex upon construction and automatically releases it when the scope is exited, which happens when the function returns. The order in which the "Resource value: " lines are printed is not determined—it depends on how the operating system schedules the threads, but each increment of resource and the corresponding output will be safely serialized due to the mutex. Hence, you will see "Resource value: 1" followed by "Resource value: 2", but the thread that executes first is not predetermined.
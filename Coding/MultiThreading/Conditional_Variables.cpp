#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <math.h>
using namespace std;

// 13. Roman to Integer

/*

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000


Roman numerals are usually written largest to smallest from left to right. 
However, the numeral for four is not IIII. Instead, the number four is written as IV. 
Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX.
There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.


Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.
Example 2:

Input: s = "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
Example 3:

Input: s = "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
 

Constraints:

1 <= s.length <= 15
s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
It is guaranteed that s is a valid roman numeral in the range [1, 3999].

*/
using namespace std::literals;
using namespace std::chrono;

// The shared data
std::string sdata;

// bool flag for predicate
bool update_progress = false;
bool completed = false;

// Mutexes
std::mutex data_mutex;
std::mutex completed_mutex;

// Conditional Variables
std::condition_variable data_cv;
std::condition_variable completed_cv;

void fetch_data(){

  for(int i = 0; i < 5; ++i){
    std::cout << "Fetcher thread waiting for data..." << std::endl;
    std::this_thread::sleep_for(2s);
    // Update sdata, then notify the progress bar thread
    std::unique_lock<std::mutex> uniq_lck(data_mutex);
    sdata += "Block" + std::to_string(i+1);
    std::cout << "Fetched data: " << sdata << std::endl;
    update_progress = true;
    uniq_lck.unlock();
    data_cv.notify_all();

  }

  std::cout << "Fetched data has ended\n";

  std::lock_guard<std::mutex> lg(completed_mutex);
  completed = true;
  completed_cv.notify_all();

}

void progress_bar()
{
  size_t len = 0;

  while(true){
    
    std::cout << "Progress bar thread waiting for data..." << std::endl;

    // Wait until there is some new data to display
    std::unique_lock<std::mutex> data_lck(data_mutex);
    data_cv.wait(data_lck, []{return update_progress;});

    len = sdata.size();
    update_progress = false;
    data_lck.unlock();

    std::cout << "Received " << len << " bytes so far\n" << std::endl;

    // Check if the download has finished
    std::unique_lock<std::mutex> compl_lck(completed_mutex);
    // Use wait_for() to avoid blocking
    if(completed_cv.wait_for(compl_lck,10ms, [] { return completed;})){
      std::cout << "Progress bar thread has ended" << std::endl;
      break;
    }
  }


void process_data(){

  std::this_thread::sleep_for(200ms);
  std::cout << "Processing thread waiting for data..." << std::endl;
  // Wait until the download is complete
  std::unique_lock<std::mutex> compl_lck(completed_mutex);
  completed_cv.wait(compl_lck, []{return completed;});
  compl_lck.unlock();
  
  std::lock_guard<std::mutex> data_lck(data_mutex);
  std::cout << "Processing sdata: " << sdata << std::endl;

}

int main(){

    std::thread fetcher(fetch_data);
    std::thread prog(progress_bar);
    std::thread processor(process_data);
    
    fetcher.join();
    prog.join();
    processor.join();

  return 0;
}



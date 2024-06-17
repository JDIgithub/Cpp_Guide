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
#include <fstream>

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



class FileManager {
public:
    static void writeTextFile(const std::string& filename, const std::string& content) {
        std::ofstream outfile(filename);
        if (outfile.is_open()) {
            outfile << content;
            outfile.close();
        } else {
            std::cerr << "Unable to open file for writing" << std::endl;
        }
    }

    static void readTextFile(const std::string& filename) {
        std::ifstream infile(filename);
        std::string line;
        if (infile.is_open()) {
            while (getline(infile, line)) {
                std::cout << line << std::endl;
            }
            infile.close();
        } else {
            std::cerr << "Unable to open file for reading" << std::endl;
        }
    }

    static void writeBinaryFile(const std::string& filename, const std::vector<int>& data) {
        std::ofstream outfile(filename, std::ios::binary);
        if (outfile.is_open()) {
            outfile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int));
            outfile.close();
        } else {
            std::cerr << "Unable to open file for writing" << std::endl;
        }
    }

    static void readBinaryFile(const std::string& filename) {
        std::ifstream infile(filename, std::ios::binary);
        if (infile.is_open()) {
            std::vector<int> data(5);
            infile.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(int));
            infile.close();
            for (int num : data) {
                std::cout << num << " ";
            }
            std::cout << std::endl;
        } else {
            std::cerr << "Unable to open file for reading" << std::endl;
        }
    }
};

int main() {
    FileManager::writeTextFile("example.txt", "Hello, World!\nThis is a simple text file.");
    FileManager::readTextFile("example.txt");

    std::vector<int> data = {10, 20, 30, 40, 50};
    FileManager::writeBinaryFile("example.bin", data);
    FileManager::readBinaryFile("example.bin");

    return 0;
}



int main() {

  // Writing into file
  std::ofstream outfile("example.txt");

  if (outfile.is_open()) {
    outfile << "Hello, World!" << std::endl;
    outfile << "This is a simple text file." << std::endl;
    outfile.close();
  } else {
    std::cerr << "Unable to open file for writing" << std::endl;
  }


  // Reading file
  std::ifstream infile("example.txt");
  std::string line;

  if (infile.is_open()) {
    while (getline(infile, line)) {
      std::cout << line << std::endl;
    }
    infile.close();
  } else {
    std::cerr << "Unable to open file for reading" << std::endl;
  }


  // Writing into Binary File
  std::ofstream outfileBin("example.bin", std::ios::binary);

  if (outfileBin.is_open()) {
    int number = 12345;
    outfileBin.write(reinterpret_cast<const char*>(&number), sizeof(number));
    outfileBin.close();
  } else {
    std::cerr << "Unable to open file for writing" << std::endl;
  }


  // Reading the Binary File
  std::ifstream infileBin("example.bin", std::ios::binary);

  if (infileBin.is_open()) {
    int number;
    infileBin.read(reinterpret_cast<char*>(&number), sizeof(number));
    infileBin.close();

    std::cout << number << std::endl;
  } else {
    std::cerr << "Unable to open file for reading" << std::endl;
  }


  return 0;
}





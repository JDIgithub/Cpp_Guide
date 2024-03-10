#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>


// Dummy hash function for demonstration purposes
std::size_t hashFileContent(const std::string& filePath) {

  std::ifstream file(filePath, std::ios::binary);
  if (!file) { 
    std::cerr << "Could not open file: " << filePath << std::endl;
    return 0;
  }
  // Compact and efficient way to read the entire contents of a file into a std::string using STL
  std::string content;
  char buffer[1024];  // For large files it is memory efficient to read in chunks (1024 bytes here)
  while (file.read(buffer, sizeof(buffer) || file.gcount())){
    content.append((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    //std::istreambuf_iterator<char>: This is an iterator that reads characters (char) from a stream buffer. 
    // It's designed to iterate over the input stream character by character. When constructed with a std::ifstream object (in this case, file), 
    // it creates an iterator that points to the beginning of the stream. The second use of std::istreambuf_iterator<char>() 
    // without an argument creates a default-constructed (end-of-stream) iterator, which serves as a sentinel value indicating the end of the stream.
  }
  return std::hash<std::string>{}(content);
}

int main() {
  std::ifstream fileList("filelist.txt");
  if(!fileList){ 
    std::cerr << "Could not open filelist.txt" << std::endl; 
    return 1;  
  }
  std::string filePath;
  std::unordered_map<std::size_t, std::vector<std::string>> fileHashes;

  while (getline(fileList, filePath)) {
    auto hash = hashFileContent(filePath);
    if (hash != 0){ fileHashes[hash].push_back(filePath); } 
  }  

  for (const auto& entry : fileHashes) {
    if (entry.second.size() > 1) { // Only print groups of duplicates
      for (const auto& path : entry.second) { std::cout << path << ","; }
      std::cout << "\b \n"; // Replace the last comma with a newline
    }
  }
  return 0;
}
#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>


struct Node{

  Node(const std::string& key, int value): m_key(key),m_value(value), m_next(nullptr) { 
  }
  std::string m_key;
  int m_value;
  Node* m_next;

};

class Graph {

private:
  std::unordered_map<std::string, std::unordered_set<std::string>> m_adjList;

public:

  void printGraph(){
    for(const auto& [vertex,edges] : m_adjList){
      std::cout << vertex << ": [ ";
      for(const auto& edge: edges){
        std::cout << edge << " ";
      }
      std::cout << "]" << std::endl;
    }


  }

  bool addVertex(std::string vertex){ 
    // Check if the vertex is already there or not
    if(m_adjList.count(vertex) == 0){
      m_adjList[vertex];
      return true;
    } else {
      return false;
    }
  }

  bool addEdge(const std::string& vertex1, const std::string& vertex2){
    if(m_adjList.count(vertex1) != 0 && m_adjList.count(vertex2) != 0){
      m_adjList.at(vertex1).insert(vertex2);  // Inserting vertex 2 into vertex1's unordered_set
      m_adjList.at(vertex2).insert(vertex1);  // And vice-versa
      return true;
    } else {
      return false;
    }
  }
  
  bool removeEdge(const std::string& vertex1, const std::string& vertex2){
    if(m_adjList.count(vertex1) != 0 && m_adjList.count(vertex2) != 0){
      m_adjList.at(vertex1).erase(vertex2);  // Inserting vertex 2 into vertex1's unordered_set
      m_adjList.at(vertex2).erase(vertex1);  // And vice-versa
      return true;
    } else {
      return false;
    }
  }

  // When removing vertex we have to remove all of the edges that D has and other vertices have to D
  bool removeVertex(const std::string& vertex){

    if(m_adjList.count(vertex) == 0){
      return false;
    } 

    for(auto otherVertex: m_adjList.at(vertex)){
      m_adjList.at(otherVertex).erase(vertex);    // Erases edges from other vertices to our vertex
    }
    m_adjList.erase(vertex);                      // Erases our vertex together with it's set of edges to other vertices
    return true;

  }

};

int main() {

  Graph *myGraph = new Graph();

  myGraph->addVertex("A");
  myGraph->addVertex("B");
  myGraph->addVertex("C");
  myGraph->addEdge("A","B");
  myGraph->addEdge("B","C");
  myGraph->addEdge("C","A");
  myGraph->printGraph();
 // myGraph->removeEdge("A","B");
  
  myGraph->removeVertex("C");
  myGraph->printGraph();




  return 0;
}



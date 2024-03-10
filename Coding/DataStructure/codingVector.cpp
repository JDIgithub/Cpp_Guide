#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>



class VersionControl {
private:
  std::vector<std::string> m_versions;

public:
  VersionControl() {}

  void saveVersion(const std::string& content) { m_versions.push_back(content); }
  std::string getVersion(int version) {
    if (version < 1 || version > m_versions.size()) { return ""; } // Version out of range
    return m_versions[version - 1];
  }
  std::vector<std::string> getHistory() { return m_versions; }

};

int main() {
    VersionControl vc;

    vc.saveVersion("Initial version");
    vc.saveVersion("Second version");
    vc.saveVersion("Third version");

    std::cout << "Version 2: " << vc.getVersion(2) << std::endl;

    std::cout << "History:" << std::endl;
    for (const auto& version : vc.getHistory()) {
        std::cout << version << std::endl;
    }

    return 0;
}





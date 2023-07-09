
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <ratio>
#include <string>
#include <unordered_map>

#include "heuristic.h"
#include "inputs.h"

namespace {
using namespace std::literals;

constexpr auto* INSTANCES_FILENAME = "instances.txt";
constexpr auto* BENCHMARK_OUTPUT_FILENAME = "cpp_execution_times.csv";
constexpr auto* TESTS_DIRNAME = "../../../tests";

using DataType = float;

auto benchmark_execution(const auto& directory,
                         const auto& instance,
                         auto runs) {
  auto elapsed_times = std::vector<double>{};

  for (size_t i = 0; i < runs; ++i) {
    auto start_time = std::chrono::high_resolution_clock::now();
    auto [jobs, number_jobs, number_machines] =
        pfsp::read_instance_data<DataType, neh::Job<DataType>>(directory /
                                                               instance);
    const auto [solution, elapsed] =
        neh::solve(std::move(jobs), number_jobs, number_machines);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(end_time - start_time);
    elapsed_times.emplace_back(duration.count());
  }
  return elapsed_times;
}


auto write_to_csv(const auto& tests_dir, const auto& execution_times) {
  auto file = std::ofstream{tests_dir / BENCHMARK_OUTPUT_FILENAME};
  for (const auto& [instance, times] : execution_times) {
    file << instance << ",";
    for (auto time : times) {
      file << time << ",";
    }
    file << "\n";
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Please provide the data dir path as a command-line argument.\n";
    return 1;
  }
  const auto data_dir = std::filesystem::absolute(argv[1]);
  const auto instances_path = data_dir / INSTANCES_FILENAME;
  const auto runs = size_t{1000};
  auto execution_times = std::unordered_map<std::string, std::vector<double>>{};

  const auto instance_file = data_dir;
  for (const auto& instance : pfsp::get_lines(instances_path.string())) {
    std::cout << "Instance name: " << instance << "\n";
    auto elapsed_times = benchmark_execution(data_dir, instance + ".txt", runs);
    execution_times[instance] = elapsed_times;
  }
  write_to_csv(data_dir / TESTS_DIRNAME, execution_times);

  return 0;
}
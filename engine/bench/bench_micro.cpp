#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "limit_order_book.hpp"

using namespace std::chrono;

int main() {
  ob::LimitOrderBook lob;

  constexpr int N = 100000;
  std::vector<double> prices;
  prices.reserve(N);
  for (int i = 0; i < N; ++i) prices.push_back(100.0 + (i % 50));

  // Benchmark upsert asks
  auto t0 = high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    lob.execute_limit(ob::Side::Ask, prices[i], 1.0);
  }
  auto t1 = high_resolution_clock::now();
  auto d_upsert = duration_cast<nanoseconds>(t1 - t0).count();

  // Benchmark market sweep with step limit
  auto t2 = high_resolution_clock::now();
  for (int i = 0; i < N / 10; ++i) {
    lob.execute_market_steps(ob::Side::Bid, 3.0, 3);
  }
  auto t3 = high_resolution_clock::now();
  auto d_sweep = duration_cast<nanoseconds>(t3 - t2).count();

  // Print simple ns/op
  double ns_per_upsert = static_cast<double>(d_upsert) / static_cast<double>(N);
  double ns_per_sweep = static_cast<double>(d_sweep) / static_cast<double>(N / 10);
  std::printf("upsert(ns/op)=%.1f\n", ns_per_upsert);
  std::printf("sweep(ns/op)=%.1f\n", ns_per_sweep);

  return 0;
}


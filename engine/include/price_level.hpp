#pragma once

#include <cstddef>

namespace ob {

class PriceLevel {
 public:
  PriceLevel() = default;
  explicit PriceLevel(double aggregateSize) : total_size_(aggregateSize) {}

  void set_total_size(double aggregateSize) { total_size_ = aggregateSize; }
  void add_size(double delta) { total_size_ += delta; }
  void remove_size(double delta) { total_size_ -= delta; }

  [[nodiscard]] double total_size() const { return total_size_; }
  [[nodiscard]] bool empty() const { return total_size_ <= 0.0; }

 private:
  double total_size_ {0.0};
};

}  


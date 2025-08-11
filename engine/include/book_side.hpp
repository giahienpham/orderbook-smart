#pragma once

#include <map>
#include <optional>
#include <vector>

#include "price_level.hpp"

namespace ob {

enum class Side { Bid, Ask };

// - Bids: high -> low
// - Asks: low -> high
class BookSide {
 public:
  explicit BookSide(Side side) : side_(side) {}

  void upsert(double price, double newSize);

  void add(double price, double delta);
  void remove(double price, double delta);

  void erase(double price);

  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::optional<double> best_price() const;
  [[nodiscard]] double total_size() const;

  // Return up to n best [price, size] pairs in book order.
  [[nodiscard]] std::vector<std::pair<double, double>> top_n(std::size_t n) const;

 private:
  Side side_ {Side::Bid};
  std::map<double, PriceLevel> levels_;
};

}  


#include "book_side.hpp"

namespace ob {

void BookSide::upsert(double price, double newSize) {
  if (newSize <= 0.0) {
    erase(price);
    return;
  }
  auto it = levels_.find(price);
  if (it == levels_.end()) {
    levels_.emplace(price, PriceLevel{newSize});
  } else {
    it->second.set_total_size(newSize);
  }
}

void BookSide::add(double price, double delta) {
  if (delta <= 0.0) return;
  auto it = levels_.find(price);
  if (it == levels_.end()) {
    levels_.emplace(price, PriceLevel{delta});
  } else {
    it->second.add_size(delta);
  }
}

void BookSide::remove(double price, double delta) {
  if (delta <= 0.0) return;
  auto it = levels_.find(price);
  if (it == levels_.end()) return;
  it->second.remove_size(delta);
  if (it->second.empty()) {
    levels_.erase(it);
  }
}

void BookSide::erase(double price) {
  auto it = levels_.find(price);
  if (it != levels_.end()) levels_.erase(it);
}

bool BookSide::empty() const { return levels_.empty(); }

std::optional<double> BookSide::best_price() const {
  if (levels_.empty()) return std::nullopt;
  if (side_ == Side::Ask) {
    return levels_.begin()->first;  // smallest price
  }
  return levels_.rbegin()->first;  // largest price
}

double BookSide::total_size() const {
  double s = 0.0;
  for (const auto& [price, lvl] : levels_) {
    (void)price;
    s += lvl.total_size();
  }
  return s;
}

std::vector<std::pair<double, double>> BookSide::top_n(std::size_t n) const {
  std::vector<std::pair<double, double>> out;
  out.reserve(n);
  if (n == 0 || levels_.empty()) return out;

  if (side_ == Side::Ask) {
    for (auto it = levels_.begin(); it != levels_.end() && out.size() < n; ++it) {
      out.emplace_back(it->first, it->second.total_size());
    }
  } else {  
    for (auto it = levels_.rbegin(); it != levels_.rend() && out.size() < n; ++it) {
      out.emplace_back(it->first, it->second.total_size());
    }
  }
  return out;
}

std::optional<BookSide::LevelView> BookSide::best_level() const {
  if (levels_.empty()) return std::nullopt;
  if (side_ == Side::Ask) {
    const auto& it = *levels_.begin();
    return LevelView{it.first, it.second.total_size()};
  }
  const auto& it = *levels_.rbegin();
  return LevelView{it.first, it.second.total_size()};
}

BookSide::ConsumeBestResult BookSide::consume_best(double qty) {
  ConsumeBestResult res{0.0, 0.0};
  if (qty <= 0.0 || levels_.empty()) return res;

  if (side_ == Side::Ask) {
    auto it = levels_.begin();
    double available = it->second.total_size();
    double take = qty < available ? qty : available;
    it->second.remove_size(take);
    res.consumed = take;
    res.notional = take * it->first;
    if (it->second.empty()) levels_.erase(it);
    return res;
  }

  auto it = std::prev(levels_.end());
  double available = it->second.total_size();
  double take = qty < available ? qty : available;
  it->second.remove_size(take);
  res.consumed = take;
  res.notional = take * it->first;
  if (it->second.empty()) levels_.erase(it);
  return res;
}

}  


#pragma once

#include <cstddef>
#include <deque>
#include "book_side.hpp"
#include "events.hpp"

namespace ob {

// aggregated by price level, no per-order FIFO queues yet.
class LimitOrderBook {
 public:
  LimitOrderBook() : bids_(Side::Bid), asks_(Side::Ask) {}

  OrderResult execute_market(Side side, double size);
  OrderResult execute_limit(Side side, double price, double size);

  // Convenience: sweep across multiple levels up to max_steps to avoid infinite loops.
  OrderResult execute_market_steps(Side side, double size, std::size_t max_steps);

  void cancel(Side side, double price, double size);

  [[nodiscard]] std::optional<double> best_bid() const { return bids_.best_price(); }
  [[nodiscard]] std::optional<double> best_ask() const { return asks_.best_price(); }

  [[nodiscard]] const BookSide& bids() const { return bids_; }
  [[nodiscard]] const BookSide& asks() const { return asks_; }

  [[nodiscard]] bool validate() const { return bids_.validate() && asks_.validate(); }

  [[nodiscard]] std::optional<double> mid_price() const {
    auto bb = bids_.best_price();
    auto ba = asks_.best_price();
    if (!bb || !ba) return std::nullopt;
    return (*bb + *ba) * 0.5;
  }
  [[nodiscard]] std::optional<double> spread() const {
    auto bb = bids_.best_price();
    auto ba = asks_.best_price();
    if (!bb || !ba) return std::nullopt;
    return *ba - *bb;
  }

  struct DepthSnapshot {
    std::vector<std::pair<double, double>> bids;  
    std::vector<std::pair<double, double>> asks;  
  };
  [[nodiscard]] DepthSnapshot snapshot(std::size_t depth) const {
    return DepthSnapshot{bids_.top_n(depth), asks_.top_n(depth)};
  }

  // Recent fills ring buffer (lite trade log)
  void set_recent_fills_capacity(std::size_t cap);
  std::vector<Fill> get_and_clear_recent_fills();

 private:
  BookSide bids_;
  BookSide asks_;
  std::deque<Fill> recent_fills_;
  std::size_t recent_capacity_ {512};
  void push_recent_fill(const Fill& f);
};

}  


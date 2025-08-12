#pragma once

#include "book_side.hpp"
#include "events.hpp"

namespace ob {

// aggregated by price level, no per-order FIFO queues yet.
class LimitOrderBook {
 public:
  LimitOrderBook() : bids_(Side::Bid), asks_(Side::Ask) {}

  OrderResult execute_market(Side side, double size);
  OrderResult execute_limit(Side side, double price, double size);

  void cancel(Side side, double price, double size);

  [[nodiscard]] std::optional<double> best_bid() const { return bids_.best_price(); }
  [[nodiscard]] std::optional<double> best_ask() const { return asks_.best_price(); }

  [[nodiscard]] const BookSide& bids() const { return bids_; }
  [[nodiscard]] const BookSide& asks() const { return asks_; }

 private:
  BookSide bids_;
  BookSide asks_;
};

}  


#include "top_of_book.hpp"

namespace ob {

void TopOfBook::update_bid(double price, double size) {
  if (size <= 0.0) {
    bid_.reset();
  } else {
    bid_ = Quote{price, size};
  }
}

void TopOfBook::update_ask(double price, double size) {
  if (size <= 0.0) {
    ask_.reset();
  } else {
    ask_ = Quote{price, size};
  }
}

std::optional<Quote> TopOfBook::best_bid() const { return bid_; }
std::optional<Quote> TopOfBook::best_ask() const { return ask_; }

std::optional<double> TopOfBook::mid_price() const {
  if (!bid_ || !ask_) return std::nullopt;
  return (bid_->price + ask_->price) * 0.5;
}

void TopOfBook::reset() {
  bid_.reset();
  ask_.reset();
}

bool TopOfBook::has_bid() const { return static_cast<bool>(bid_); }
bool TopOfBook::has_ask() const { return static_cast<bool>(ask_); }

TopOfBookSnapshot TopOfBook::snapshot() const {
  TopOfBookSnapshot s;
  s.bid = bid_;
  s.ask = ask_;
  s.mid = mid_price();
  return s;
}

}  // namespace ob


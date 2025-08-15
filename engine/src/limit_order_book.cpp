#include "limit_order_book.hpp"

namespace ob {

OrderResult LimitOrderBook::execute_market(Side side, double size) {
  OrderResult res{};
  double remaining = size;
  while (remaining > 0.0) {
    auto bl = (side == Side::Bid) ? asks_.best_level() : bids_.best_level();
    if (!bl.has_value()) break;
    double to_take = remaining;
    auto take_res = (side == Side::Bid)
                        ? asks_.consume_best(to_take)
                        : bids_.consume_best(to_take);
    if (take_res.consumed > 0.0) {
      Fill f{take_res.price, take_res.consumed};
      res.total_filled += take_res.consumed;
      res.notional += take_res.notional;
      res.fills.push_back(f);
      push_recent_fill(f);
    }
    remaining -= take_res.consumed;
    if (take_res.consumed <= 0.0) break;
  }
  return res;
}

OrderResult LimitOrderBook::execute_market_steps(Side side, double size, std::size_t max_steps) {
  OrderResult res{};
  double remaining = size;
  std::size_t steps = 0;
  while (remaining > 0.0 && steps < max_steps) {
    auto bl = (side == Side::Bid) ? asks_.best_level() : bids_.best_level();
    if (!bl.has_value()) break;
    double to_take = remaining;
    auto take_res = (side == Side::Bid)
                        ? asks_.consume_best(to_take)
                        : bids_.consume_best(to_take);
    if (take_res.consumed > 0.0) {
      Fill f{take_res.price, take_res.consumed};
      res.total_filled += take_res.consumed;
      res.notional += take_res.notional;
      res.fills.push_back(f);
      push_recent_fill(f);
    }
    remaining -= take_res.consumed;
    if (take_res.consumed <= 0.0) break;
    ++steps;
  }
  return res;
}

OrderResult LimitOrderBook::execute_limit(Side side, double price, double size) {
  OrderResult res{};
  double remaining = size;

  // If crosses, fill against opposite side within limit price.
  while (remaining > 0.0) {
    auto bestOpp = (side == Side::Bid) ? asks_.best_level() : bids_.best_level();
    if (!bestOpp.has_value()) break;
    bool crosses = (side == Side::Bid) ? (bestOpp->price <= price) : (bestOpp->price >= price);
    if (!crosses) break;
    double to_take = remaining;
    auto take_res = (side == Side::Bid)
                        ? asks_.consume_best(to_take)
                        : bids_.consume_best(to_take);
    if (take_res.consumed > 0.0) {
      Fill f{take_res.price, take_res.consumed};
      res.total_filled += take_res.consumed;
      res.notional += take_res.notional;
      res.fills.push_back(f);
      push_recent_fill(f);
    }
    remaining -= take_res.consumed;
    if (take_res.consumed <= 0.0) break;
  }

  // Remainder posts at its limit price.
  if (remaining > 0.0) {
    if (side == Side::Bid) {
      bids_.add(price, remaining);
    } else {
      asks_.add(price, remaining);
    }
  }
  return res;
}

void LimitOrderBook::cancel(Side side, double price, double size) {
  if (side == Side::Bid) {
    bids_.remove(price, size);
  } else {
    asks_.remove(price, size);
  }
}

void LimitOrderBook::set_recent_fills_capacity(std::size_t cap) {
  recent_capacity_ = cap == 0 ? 1 : cap;
  while (recent_fills_.size() > recent_capacity_) recent_fills_.pop_front();
}

std::vector<Fill> LimitOrderBook::get_and_clear_recent_fills() {
  std::vector<Fill> out;
  out.reserve(recent_fills_.size());
  while (!recent_fills_.empty()) {
    out.push_back(recent_fills_.front());
    recent_fills_.pop_front();
  }
  return out;
}

void LimitOrderBook::push_recent_fill(const Fill& f) {
  if (recent_capacity_ == 0) return;
  if (recent_fills_.size() >= recent_capacity_) recent_fills_.pop_front();
  recent_fills_.push_back(f);
}

}  


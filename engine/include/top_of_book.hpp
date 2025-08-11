#pragma once

#include <optional>

namespace ob {

struct Quote {
  double price {0.0};
  double size {0.0};
};

struct TopOfBookSnapshot {
  std::optional<Quote> bid;
  std::optional<Quote> ask;
  std::optional<double> mid;
};

class TopOfBook {
 public:
  void update_bid(double price, double size);
  void update_ask(double price, double size);
  void reset();

  [[nodiscard]] bool has_bid() const;
  [[nodiscard]] bool has_ask() const;

  [[nodiscard]] std::optional<Quote> best_bid() const;
  [[nodiscard]] std::optional<Quote> best_ask() const;
  [[nodiscard]] std::optional<double> mid_price() const;
  [[nodiscard]] TopOfBookSnapshot snapshot() const;

 private:
  std::optional<Quote> bid_ {};
  std::optional<Quote> ask_ {};
};

}  // namespace ob


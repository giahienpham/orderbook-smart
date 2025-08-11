#pragma once

#include <optional>

namespace ob {

struct Quote {
  double price {0.0};
  double size {0.0};
};

class TopOfBook {
 public:
  void update_bid(double price, double size);
  void update_ask(double price, double size);

  [[nodiscard]] std::optional<Quote> best_bid() const;
  [[nodiscard]] std::optional<Quote> best_ask() const;
  [[nodiscard]] std::optional<double> mid_price() const;

 private:
  std::optional<Quote> bid_ {};
  std::optional<Quote> ask_ {};
};

}  // namespace ob


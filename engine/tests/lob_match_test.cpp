#include <gtest/gtest.h>

#include "limit_order_book.hpp"

using ob::LimitOrderBook;
using ob::Side;

TEST(LOBMatch, MarketConsumesOppositeBest) {
  LimitOrderBook lob;

  lob.execute_limit(Side::Ask, 100.0, 1.0);
  lob.execute_limit(Side::Ask, 101.0, 2.0);

  auto res = lob.execute_market(Side::Bid, 1.2); 
  EXPECT_DOUBLE_EQ(res.total_filled, 1.2);
  // consume 1.0@100, then 0.2@101 → 100 erased, best ask is now 101.
  ASSERT_TRUE(lob.best_ask().has_value());
  EXPECT_DOUBLE_EQ(lob.best_ask().value(), 101.0);
}

TEST(LOBMatch, LimitCrossesThenPostsRemainder) {
  LimitOrderBook lob;
  lob.execute_limit(Side::Ask, 101.0, 2.0);

  auto res = lob.execute_limit(Side::Bid, 102.0, 3.0);  
  EXPECT_DOUBLE_EQ(res.total_filled, 2.0);
  // all asks at 101 consumed
  EXPECT_FALSE(lob.best_ask().has_value());
}


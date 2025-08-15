#include <gtest/gtest.h>

#include "limit_order_book.hpp"

using ob::LimitOrderBook;
using ob::Side;

TEST(LOBRecentFills, BasicGetAndClear) {
  LimitOrderBook lob;
  lob.execute_limit(Side::Ask, 100.0, 1.0);
  lob.execute_limit(Side::Ask, 101.0, 1.0);

  auto res = lob.execute_market(Side::Bid, 1.5);  // 1.0@100 + 0.5@101
  ASSERT_EQ(res.fills.size(), 2u);

  auto recent = lob.get_and_clear_recent_fills();
  ASSERT_EQ(recent.size(), 2u);
  EXPECT_NEAR(recent[0].price, 100.0, 1e-12);
  EXPECT_NEAR(recent[0].size, 1.0, 1e-12);
  EXPECT_NEAR(recent[1].price, 101.0, 1e-12);
  EXPECT_NEAR(recent[1].size, 0.5, 1e-12);

  // After clear, buffer is empty
  auto recent2 = lob.get_and_clear_recent_fills();
  EXPECT_TRUE(recent2.empty());
}

TEST(LOBRecentFills, CapacityEviction) {
  LimitOrderBook lob;
  lob.set_recent_fills_capacity(2);

  lob.execute_limit(Side::Ask, 100.0, 1.0);
  lob.execute_limit(Side::Ask, 101.0, 1.0);
  lob.execute_limit(Side::Ask, 102.0, 1.0);

  // Create three fills, only last two should remain
  (void)lob.execute_market(Side::Bid, 0.5);  // 0.5@100
  (void)lob.execute_market(Side::Bid, 0.5);  // 0.5@100 (erases 100), or 0.5@101 depending on previous
  (void)lob.execute_market(Side::Bid, 0.5);

  auto recent = lob.get_and_clear_recent_fills();
  ASSERT_EQ(recent.size(), 2u);
  // Just assert monotonic non-decreasing prices for asks being consumed
  EXPECT_LE(recent[0].price, recent[1].price);
}


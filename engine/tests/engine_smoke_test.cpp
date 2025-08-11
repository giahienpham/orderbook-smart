#include <gtest/gtest.h>

#include "top_of_book.hpp"

TEST(EngineSmoke, MidPrice) {
  ob::TopOfBook tob;
  EXPECT_FALSE(tob.mid_price().has_value());
  tob.update_bid(99.0, 1.0);
  EXPECT_FALSE(tob.mid_price().has_value());
  tob.update_ask(101.0, 1.0);
  ASSERT_TRUE(tob.mid_price().has_value());
  EXPECT_DOUBLE_EQ(*tob.mid_price(), 100.0);
}

TEST(EngineSmoke, ResetAndSnapshot) {
  ob::TopOfBook tob;
  EXPECT_FALSE(tob.has_bid());
  EXPECT_FALSE(tob.has_ask());

  tob.update_bid(100.0, 2.0);
  tob.update_ask(102.0, 1.5);
  EXPECT_TRUE(tob.has_bid());
  EXPECT_TRUE(tob.has_ask());

  auto snap = tob.snapshot();
  ASSERT_TRUE(snap.bid.has_value());
  ASSERT_TRUE(snap.ask.has_value());
  ASSERT_TRUE(snap.mid.has_value());
  EXPECT_DOUBLE_EQ(snap.bid->price, 100.0);
  EXPECT_DOUBLE_EQ(snap.ask->price, 102.0);

  tob.reset();
  EXPECT_FALSE(tob.has_bid());
  EXPECT_FALSE(tob.has_ask());
  EXPECT_FALSE(tob.mid_price().has_value());
}


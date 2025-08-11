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


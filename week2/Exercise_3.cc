// Code Written by Hayden Shaddix 
// Exercise 3 

#include <iostream>
#include <string>

// Function to determine the winner of rock, paper, scissors
std::string rockPaperScissors(const std::string& player1Choice, const std::string& player2Choice) {
    if (player1Choice == player2Choice) {
        return "Draw";
    } else if ((player1Choice == "rock" && player2Choice == "scissors") ||
               (player1Choice == "scissors" && player2Choice == "paper") ||
               (player1Choice == "paper" && player2Choice == "rock")) {
        return "Player 1 wins!";
    } else {
        return "Player 2 wins!";
    }
}

int main() {
    // Simulating player choices
    std::string player1Choice, player2Choice;

    // Input for both players
    std::cout << "Player 1, enter your choice (rock, paper, scissors): ";
    std::cin >> player1Choice;
    std::cout << "Player 2, enter your choice (rock, paper, scissors): ";
    std::cin >> player2Choice;

    // Determine and display the result
    std::string result = rockPaperScissors(player1Choice, player2Choice);
    std::cout << result << std::endl;

    return 0;
}

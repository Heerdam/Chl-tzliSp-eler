
#include "../include/dfsph.hpp"

#include <imgui.h>
#include <raylib.h>
#include "../include/imgui_impl_raylib.h"

int main(void)
{
    InitWindow(2000, 1500, "DFSPH 2D");

    while (!WindowShouldClose())
    {
        BeginDrawing();
            ClearBackground(RAYWHITE);
            
        EndDrawing();
    }

    CloseWindow();

    return 0;
}


/*
#include <bitset>
#include <iostream>
#include <cmath>


int main() {

    return 0;
}
*/



using Luxor

function drawmatrix(A::Matrix;
    cellsize=(10, 10))
    table = Table(size(A)..., cellsize...)
    used = Set()
    for i in CartesianIndices(A)
        r, c = Tuple(i)
        if A[r, c] ∈ used
            sethue("orange")
        else
            sethue("purple")
            push!(used, A[r, c])
        end
        text(string(A[r, c]), table[r, c],
            halign=:center,
            valign=:middle)
        sethue("white")
        box(table, r, c, :stroke)
    end
end

function draw_matrix()
    A = rand(1:99, 5, 8)
    @draw begin
        background("black")
        fontsize(30)
        setline(0.5)
        sethue("white")
        drawmatrix(A, cellsize=10 .* size(A))
    end
end

itworks() = print("It works !")




struct MatrixChunks
    X::Matrix{<:AbstractFloat}
    n::Int  # size of matrix
end

function MatrixChunks(X::Matrix{<:AbstractFloat})
    (n1, n2) = size(X)
    if n1 != n2
        error("Matrix should be square")
    end
    return MatrixChunks(X, n1)
end

function Base.iterate(iter::MatrixChunks)
    element = view(iter.X, 1:2, 1:2)
    return (element, (1, 1))
end

function Base.iterate(iter::MatrixChunks, state)
    if state[1] + 1 < iter.n
        x = state[1] + 1
        y = state[2]
        else
        if state[2] + 1 < iter.n
            x = 1
            y = state[2] + 1
        else
        return nothing
        end
    end

        element = view(iter.X, y:y + 1, x:x + 1)

    return (element, (x, y))
end



function iterate_iso_field(a::Matrix{<:AbstractFloat})
    for y = 1:size(a, 1) - 1
        for x = 1:size(a, 2) - 1
            s = view(a, y:y + 1, x:x + 1)
            display((x, y))
            display(s)
        end
    end
end

    function generate_iso_field()
    a = randn(4, 4)

    display(a)

    iterate_iso_field(a)

    println()

    for n in MatrixChunks(a)
        display(n)
    end
end
